import gc
import os
import random
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"


def _logit(x):
    # Clamp x just below 1 to avoid infinite logits.
    return -np.log((1 / (np.minimum(x, 1 - 1e-7) + 1e-8)) - 1)


def _fold_long_path(model_name, split_type, k, n_splits, batch_size, dim, resample,
                    use_demo=False, use_poli=True):
    safe = model_name.replace('/', '_')
    base = f'{safe}_{split_type}_{k}_{n_splits}_{batch_size}_{dim}__resample{resample}'
    if use_demo is False and use_poli is True:
        return PREDICTIONS_DIR / f'{base}_long.parquet'
    return PREDICTIONS_DIR / f'{base}_{use_demo}_{use_poli}_long.parquet'


def _out_path(model_name, split_type, n_splits, batch_size, dim, resample,
              use_demo=False, use_poli=True):
    safe = model_name.replace('/', '_')
    base = f'{safe}_{split_type}_{n_splits}_{batch_size}_{dim}__resample{resample}'
    if use_demo is False and use_poli is True:
        return PREDICTIONS_DIR / f'{base}_varyear.parquet'
    return PREDICTIONS_DIR / f'{base}_{use_demo}_{use_poli}_varyear.parquet'


def _build_lookups():
    """Return (df_obs, df_w):

    df_obs: per-(yearid_id, question_id) observation — lean frame with
            variable, year, yearid, binarized. Only rows where binarized
            is present.
    df_w:   per-yearid_id sampling weight (`wtssall`).
    """
    slim_path = DATA_DIR / 'df_analysis_slim.pkl'
    src = slim_path if slim_path.exists() else DATA_DIR / 'df_analysis.pkl'
    print(f'Loading {src.name} ...')
    df_analysis = pd.read_pickle(src)
    df_analysis = df_analysis.drop_duplicates(['yearid_id', 'question_id'])

    df_obs = (
        df_analysis[['yearid_id', 'question_id', 'variable', 'module', 'year', 'yearid', 'binarized']]
        .dropna(subset=['binarized'])
        .copy()
    )
    # module is NaN for ~245 (var, year) pairs — fill so groupby keeps them.
    df_obs['module'] = df_obs['module'].fillna('__nomodule__').astype(str)
    df_obs['yearid_id'] = df_obs['yearid_id'].astype('int32')
    df_obs['question_id'] = df_obs['question_id'].astype('int32')
    df_obs['year'] = df_obs['year'].astype('int32')
    df_obs['binarized'] = df_obs['binarized'].astype('float32')
    print(f'  df_obs: {len(df_obs):,} observed (yearid_id, question_id) rows, '
          f'{df_obs["variable"].nunique()} variables, {df_obs["year"].nunique()} years')

    # Sampling weight lookup: (year, id) -> wtssall, then collapse to yearid_id
    gss_df = pd.read_parquet(DATA_DIR / 'gss_df_merged.parquet')
    w_key = (
        df_analysis[['yearid_id', 'yearid', 'year']]
        .drop_duplicates()
        .assign(id=lambda d: (d['yearid'] % 10000).astype('int32'))
    )
    # wtssall covers 1972-2018; 2021 switched to wtssnrps (web mode).
    # Coalesce so every yearid_id gets the GSS-official weight for its year.
    df_w = w_key.merge(
        gss_df[['year', 'id', 'wtssall', 'wtssnrps']].drop_duplicates(),
        on=['year', 'id'], how='left',
    )
    df_w['weight'] = df_w['wtssall'].fillna(df_w['wtssnrps']).astype('float32')
    df_w = df_w[['yearid_id', 'weight']].rename(columns={'weight': 'wtssall'})
    df_w['yearid_id'] = df_w['yearid_id'].astype('int32')

    n_missing_w = df_w['wtssall'].isna().sum()
    if n_missing_w > 0:
        affected_years = (
            df_w.merge(df_obs[['yearid_id', 'year']].drop_duplicates(), on='yearid_id', how='left')
            .loc[lambda d: d['wtssall'].isna(), 'year']
            .value_counts()
            .to_dict()
        )
        print(f'  WARNING: {n_missing_w:,} yearid_ids missing weight; by year: {affected_years}')
    else:
        print(f'  All {len(df_w):,} yearid_ids have weight (wtssall / wtssnrps for 2021).')

    return df_obs, df_w


def _process_fold(long_path, k, df_obs, df_w):
    """Load one fold, aggregate to (variable, year) on train & val, return
    (train_agg, val_agg) as DataFrames with columns:
       variable, year, pred_logit_wavg, obs_wavg, n
    Returns (None, None) if the fold file is missing or empty.
    """
    if not long_path.exists():
        return None, None

    resp = f'response_{k}'
    val = f'validation_{k}'
    long = pd.read_parquet(long_path, columns=['yearid_id', 'question_id', resp, val])
    long['yearid_id'] = long['yearid_id'].astype('int32')
    long['question_id'] = long['question_id'].astype('int32')
    long[resp] = long[resp].astype('float32')
    long[val] = long[val].astype('int8')

    # Keep only rows with observations (inner join on df_obs) and training/val
    # (drop validation==2).
    df = long.merge(df_obs, on=['yearid_id', 'question_id'], how='inner')
    del long
    gc.collect()
    df = df.loc[df[val] != 2]

    # Attach weights
    df = df.merge(df_w, on='yearid_id', how='left')
    df = df.loc[df['wtssall'].notna() & (df['wtssall'] > 0)]

    # Logit of raw prediction; for models whose predictions leave [0, 1]
    # (MF raw ALS scores) aggregate the clipped probabilities instead.
    resp_vals = df[resp]
    if float(resp_vals.min()) < 0.0 or float(resp_vals.max()) > 1.0:
        print(f'  fold {k}: predictions outside [0,1] '
              f'(min {resp_vals.min():.3f}, max {resp_vals.max():.3f}) '
              f'- aggregating clipped probabilities')
        df['pred_logit'] = resp_vals.clip(0.0, 1.0).astype('float32')
    else:
        df['pred_logit'] = _logit(resp_vals).astype('float32')

    def _agg(sub):
        cols = ['variable', 'module', 'year', 'pred_logit_wavg', 'obs_wavg', 'sw', 'n']
        if len(sub) == 0:
            return pd.DataFrame(columns=cols)
        w = sub['wtssall'].values
        pl = sub['pred_logit'].values
        ob = sub['binarized'].values
        sub = sub.assign(_wp=w * pl, _wo=w * ob, _w=w)
        # module is functionally determined by variable — group includes it
        # so the column survives without needing `first`.
        agg = sub.groupby(['variable', 'module', 'year'], observed=True).agg(
            _wp=('_wp', 'sum'),
            _wo=('_wo', 'sum'),
            _w=('_w', 'sum'),
            n=('binarized', 'size'),
        ).reset_index()
        agg = agg.loc[agg['_w'] > 0].copy()
        agg['pred_logit_wavg'] = agg['_wp'] / agg['_w']
        agg['obs_wavg'] = agg['_wo'] / agg['_w']
        agg = agg.rename(columns={'_w': 'sw'})  # sum of weights
        return agg[cols]

    train_agg = _agg(df.loc[df[val] == 0])
    val_agg = _agg(df.loc[df[val] == 1])

    del df
    gc.collect()
    return train_agg, val_agg


def run(model_name='mf', split_type='partial', n_splits=10, resample=1,
        batch_size=128, dim=50, use_demo=False, use_poli=True, k_only=-1):
    """Build (variable, year) GLM predictions per fold for one (model, split_type)."""
    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    np.random.seed(42)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Check that at least the first fold exists; otherwise skip gracefully.
    first = _fold_long_path(model_name, split_type, 0, n_splits, batch_size, dim,
                            resample, use_demo, use_poli)
    if not first.exists():
        print(f'SKIP: no fold 0 at {first}')
        return

    df_obs, df_w = _build_lookups()

    rows = []
    folds = [k_only] if k_only >= 0 else range(n_splits)
    for k in folds:
        long_path = _fold_long_path(model_name, split_type, k, n_splits, batch_size, dim,
                                    resample, use_demo, use_poli)
        print(f'\n--- Fold {k}: {long_path.name} ---')
        train_agg, val_agg = _process_fold(long_path, k, df_obs, df_w)
        if train_agg is None:
            print(f'  missing file, skip fold {k}')
            continue

        if len(train_agg) < 2:
            print(f'  train_agg too small ({len(train_agg)}), skip fold {k}')
            continue
        if len(val_agg) == 0:
            print(f'  val_agg empty, skip fold {k}')
            continue

        # Held-out (var, year) is only valid if the model has seen the
        # relevant unit at another year during training:
        #   partial: same variable (variable-level holdout)
        #   module:  same module  (module-level holdout; Core is always in
        #            training, so Core variables pass trivially)
        if split_type in ('partial', 'module'):
            key = 'variable' if split_type == 'partial' else 'module'
            train_keys = set(train_agg[key].unique())
            n_before = len(val_agg)
            val_agg = val_agg.loc[val_agg[key].isin(train_keys)].copy()
            n_dropped = n_before - len(val_agg)
            if n_dropped > 0:
                print(f'  {split_type}: dropped {n_dropped:,} val (var, year) rows '
                      f'whose {key} is absent from training')
            if len(val_agg) == 0:
                print(f'  val_agg empty after {split_type} filter, skip fold {k}')
                continue

        # Fit GLM on training aggregate
        try:
            glm = sm.GLM(
                train_agg['obs_wavg'].values,
                sm.add_constant(train_agg['pred_logit_wavg'].values),
                family=sm.families.Binomial(),
            ).fit()
        except Exception as e:
            print(f'  GLM fit failed: {e}; skip fold {k}')
            continue
        b0, b1 = glm.params[0], glm.params[1]
        print(f'  GLM fit on {len(train_agg):,} (var, year) train groups: '
              f'const={b0:.4f}, slope={b1:.4f}')

        # Predict on validation aggregate
        val_agg = val_agg.copy()
        val_agg['GLM_predicted'] = glm.predict(
            sm.add_constant(val_agg['pred_logit_wavg'].values, has_constant='add')
        ).astype('float32')
        val_agg['fold'] = np.int8(k)
        val_agg['glm_const'] = np.float32(b0)
        val_agg['glm_slope'] = np.float32(b1)
        val_agg = val_agg.rename(columns={'n': 'n_val', 'sw': 'sw_val'})
        # Attach train group size + sum-of-weights per (variable, year)
        val_agg = val_agg.merge(
            train_agg[['variable', 'year', 'n', 'sw']].rename(
                columns={'n': 'n_train', 'sw': 'sw_train'}),
            on=['variable', 'year'], how='left',
        )
        val_agg['n_train'] = val_agg['n_train'].fillna(0).astype('int32')
        val_agg['n_val'] = val_agg['n_val'].astype('int32')
        val_agg['sw_train'] = val_agg['sw_train'].fillna(0).astype('float32')
        val_agg['sw_val'] = val_agg['sw_val'].astype('float32')
        print(f'  val_agg: {len(val_agg):,} (var, year) rows; '
              f'GLM_predicted in [{val_agg["GLM_predicted"].min():.4f}, '
              f'{val_agg["GLM_predicted"].max():.4f}]')

        rows.append(val_agg[[
            'variable', 'year', 'fold',
            'pred_logit_wavg', 'obs_wavg', 'GLM_predicted',
            'glm_const', 'glm_slope',
            'n_train', 'n_val', 'sw_train', 'sw_val',
        ]])

        del train_agg, val_agg
        gc.collect()

    if not rows:
        print('No folds produced output; nothing to save.')
        return

    per_fold = pd.concat(rows, ignore_index=True)

    # Collapse fold-level rows to one row per (variable, year) via
    # sw_val-weighted means. For partial/total/module each (var, year)
    # has exactly one fold, so this is a no-op; for impute it averages
    # across the ~10 folds that each contribute distinct respondents.
    per_fold['_wp']  = per_fold['pred_logit_wavg'] * per_fold['sw_val']
    per_fold['_wg']  = per_fold['GLM_predicted']   * per_fold['sw_val']
    per_fold['_wo']  = per_fold['obs_wavg']        * per_fold['sw_val']
    out = per_fold.groupby(['variable', 'year'], observed=True).agg(
        pred_logit_wavg=('_wp', 'sum'),
        GLM_predicted=('_wg', 'sum'),
        obs_wavg=('_wo', 'sum'),
        sw_val=('sw_val', 'sum'),
        n_val=('n_val', 'sum'),
        n_folds=('fold', 'nunique'),
    ).reset_index()
    out['pred_logit_wavg'] = (out['pred_logit_wavg'] / out['sw_val']).astype('float32')
    out['GLM_predicted']   = (out['GLM_predicted']   / out['sw_val']).astype('float32')
    out['obs_wavg']        = (out['obs_wavg']        / out['sw_val']).astype('float32')
    out['sw_val']          = out['sw_val'].astype('float32')
    out['n_val']           = out['n_val'].astype('int32')
    out['n_folds']         = out['n_folds'].astype('int8')

    out_path = _out_path(model_name, split_type, n_splits, batch_size, dim,
                        resample, use_demo, use_poli)
    out.to_parquet(out_path)
    print(f'\nSaved -> {out_path} (shape: {out.shape}, '
          f'unique (var, year): {len(out):,})')


if __name__ == '__main__':
    fire.Fire()
