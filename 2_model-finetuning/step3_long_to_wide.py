import gc
import os
import pickle
import random
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from tqdm.auto import tqdm
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"


def wide(model_name='chavinlo/alpaca-native', split_type='partial', n_splits=10,
         resample=1, batch_size=128, dim=50, use_demo=False, use_poli=True):
    """Combine per-fold long predictions into a single wide parquet."""

    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    np.random.seed(42)

    safe_name = model_name.replace('/', '_')

    # Load df_analysis for variable mapping and observed values
    from utils import load_data
    df_analysis, _, _ = load_data(split_type=split_type, k=0, n_splits=n_splits,
                                   resample=resample, use_demo=use_demo, use_poli=use_poli)

    # Load all fold predictions into one DataFrame
    print('Loading per-fold predictions...')
    if use_demo is False and use_poli is True:
        base_pattern = f'{safe_name}_{split_type}_{{k}}_{n_splits}_{batch_size}_{dim}__resample{resample}_long.parquet'
    else:
        base_pattern = f'{safe_name}_{split_type}_{{k}}_{n_splits}_{batch_size}_{dim}__resample{resample}_{use_demo}_{use_poli}_long.parquet'

    # Load fold 0 as base
    path_0 = PREDICTIONS_DIR / base_pattern.format(k=0)
    df_list = pd.read_parquet(path_0)
    print(f'  Fold 0: {len(df_list)} rows')

    # Merge remaining folds, verifying that every fold's (yearid_id, question_id)
    # rows align with fold 0.
    base_yid = df_list['yearid_id'].to_numpy()
    base_qid = df_list['question_id'].to_numpy()
    for k in range(1, n_splits):
        path_k = PREDICTIONS_DIR / base_pattern.format(k=k)
        fold_k = pd.read_parquet(
            path_k, columns=['yearid_id', 'question_id', f'response_{k}', f'validation_{k}'])
        if len(fold_k) != len(df_list) \
                or not np.array_equal(fold_k['question_id'].to_numpy(), base_qid) \
                or not np.array_equal(fold_k['yearid_id'].to_numpy(), base_yid):
            raise ValueError(
                f'Fold {k} rows do not align with fold 0 '
                f'({len(fold_k)} vs {len(df_list)} rows). All *_long.parquet '
                f'folds must be produced by the same step2_predict.py settings '
                f'(in particular, the same --val_only). Regenerate the '
                f'mismatched folds before building the wide file.')
        df_list[f'response_{k}'] = fold_k[f'response_{k}']
        df_list[f'validation_{k}'] = fold_k[f'validation_{k}']
        del fold_k
        gc.collect()
        print(f'  Fold {k}: merged (aligned)')
    del base_yid, base_qid

    # Merge observed binarized values
    df_analysis_bin = df_analysis[['yearid_id', 'question_id', 'binarized']].drop_duplicates()
    df_list_reg = df_list.merge(df_analysis_bin, on=['yearid_id', 'question_id'], how='right')
    df_list = df_list.merge(df_analysis_bin, on=['yearid_id', 'question_id'], how='left')

    # For total split: extend validation flags to all year_orders of held-out questions
    if split_type == 'total':
        for k in tqdm(range(n_splits), desc='Extending total validation'):
            qids = set(df_list.loc[df_list[f'validation_{k}'] == 1, 'question_id'])
            df_list[f'validation_{k}'] = df_list['question_id'].isin(qids).astype(int)

    # Switch to out-of-sample predictions via validation flags
    print('Switching to out-of-sample predictions...')
    df_list['response'] = df_list['response_0']

    # Helper: logit transform
    def _logit(x):
        # Clamp x just below 1 to avoid infinite logits.
        return -np.log((1 / (np.minimum(x, 1 - 1e-7) + 1e-8)) - 1)

    # MF emits raw ALS dot products, not probabilities, so the logit-based
    # rescale variants are skipped whenever predictions leave [0, 1].
    in_prob_scale = (float(df_list['response_0'].min()) >= 0.0
                     and float(df_list['response_0'].max()) <= 1.0)
    if not in_prob_scale:
        print('Predictions outside [0, 1] (e.g. MF raw scores): '
              'skipping the *_rescale_logit* calibration columns.')

    # --- Fold 0: fit the rescale models on training data ---
    train_mask_0 = df_list_reg['validation_0'] == 0
    train_agg_0 = df_list_reg.loc[train_mask_0].groupby(['year_order', 'question_id']).agg(
        pred=('response_0', 'mean'), obs=('binarized', 'mean')
    ).dropna()

    # 1) _rescale: LR on raw pred
    df_list_reg['intercept'] = 1
    reg_lr = LinearRegression(fit_intercept=False).fit(
        df_list_reg.loc[train_mask_0].groupby(['year_order', 'question_id'])[['response_0', 'intercept']].mean(),
        df_list_reg.loc[train_mask_0].groupby(['year_order', 'question_id'])['binarized'].mean()
    )
    df_list['response_rescale'] = df_list['response_0'] * reg_lr.coef_[0] + reg_lr.coef_[1]

    # 2) _rescale_logit: LR on logit(pred)
    if in_prob_scale:
        df_list_reg['response_0_logit'] = _logit(df_list_reg['response_0'])
        reg_lr_logit = LinearRegression(fit_intercept=False).fit(
            df_list_reg.loc[train_mask_0].groupby(['year_order', 'question_id'])[['response_0_logit', 'intercept']].mean(),
            df_list_reg.loc[train_mask_0].groupby(['year_order', 'question_id'])['binarized'].mean()
        )
        df_list['response_rescale_logit'] = _logit(df_list['response_0']) * reg_lr_logit.coef_[0] + reg_lr_logit.coef_[1]

    # 3) _rescale_glm: GLM(Binomial) on raw pred → sigmoid(β₀ + β₁ * pred)
    glm_raw = sm.GLM(train_agg_0['obs'], sm.add_constant(train_agg_0['pred']),
                     family=sm.families.Binomial()).fit()
    df_list['response_rescale_glm'] = glm_raw.predict(sm.add_constant(df_list['response_0']))

    # 4) _rescale_logit_glm: GLM(Binomial) on logit(pred) → sigmoid(β₀ + β₁ * logit(pred))
    if in_prob_scale:
        train_agg_0['pred_logit'] = _logit(train_agg_0['pred'])
        glm_logit = sm.GLM(train_agg_0['obs'], sm.add_constant(train_agg_0['pred_logit']),
                           family=sm.families.Binomial()).fit()
        df_list['response_rescale_logit_glm'] = glm_logit.predict(sm.add_constant(_logit(df_list['response_0'])))

    if in_prob_scale:
        RESCALE_COLS = ['response_rescale', 'response_rescale_logit', 'response_rescale_glm', 'response_rescale_logit_glm']
    else:
        RESCALE_COLS = ['response_rescale', 'response_rescale_glm']

    for k in tqdm(range(1, n_splits), desc='Fold switching'):
        gc.collect()
        mask_val = df_list[f'validation_{k}'] == 1
        mask_avg = df_list[f'validation_{k}'] == 2
        resp_k = f'response_{k}'

        # Switch raw response
        df_list.loc[mask_val, 'response'] = df_list.loc[mask_val, resp_k]
        df_list.loc[mask_avg, 'response'] = df_list.loc[mask_avg, 'response'] + df_list.loc[mask_avg, resp_k]

        # Fit all 4 models on this fold's training data
        train_mask_k = df_list_reg[f'validation_{k}'] == 0
        train_agg_k = df_list_reg.loc[train_mask_k].groupby(['year_order', 'question_id']).agg(
            pred=(resp_k, 'mean'), obs=('binarized', 'mean')
        ).dropna()

        # 1) LR raw
        reg_lr = LinearRegression(fit_intercept=False).fit(
            df_list_reg.loc[train_mask_k].groupby(['year_order', 'question_id'])[[resp_k, 'intercept']].mean(),
            df_list_reg.loc[train_mask_k].groupby(['year_order', 'question_id'])['binarized'].mean()
        )
        df_list.loc[mask_val, 'response_rescale'] = df_list.loc[mask_val, resp_k] * reg_lr.coef_[0] + reg_lr.coef_[1]
        df_list.loc[mask_avg, 'response_rescale'] += df_list.loc[mask_avg, resp_k] * reg_lr.coef_[0] + reg_lr.coef_[1]

        # 2) LR logit
        if in_prob_scale:
            df_list_reg[f'{resp_k}_logit'] = _logit(df_list_reg[resp_k])
            reg_lr_logit = LinearRegression(fit_intercept=False).fit(
                df_list_reg.loc[train_mask_k].groupby(['year_order', 'question_id'])[[f'{resp_k}_logit', 'intercept']].mean(),
                df_list_reg.loc[train_mask_k].groupby(['year_order', 'question_id'])['binarized'].mean()
            )
            logit_k = _logit(df_list[resp_k])
            df_list.loc[mask_val, 'response_rescale_logit'] = logit_k[mask_val] * reg_lr_logit.coef_[0] + reg_lr_logit.coef_[1]
            df_list.loc[mask_avg, 'response_rescale_logit'] += logit_k[mask_avg] * reg_lr_logit.coef_[0] + reg_lr_logit.coef_[1]

        # 3) GLM raw
        glm_raw = sm.GLM(train_agg_k['obs'], sm.add_constant(train_agg_k['pred']),
                         family=sm.families.Binomial()).fit()
        df_list.loc[mask_val, 'response_rescale_glm'] = glm_raw.predict(sm.add_constant(df_list.loc[mask_val, resp_k]))
        df_list.loc[mask_avg, 'response_rescale_glm'] += glm_raw.predict(sm.add_constant(df_list.loc[mask_avg, resp_k]))

        # 4) GLM logit
        if in_prob_scale:
            train_agg_k['pred_logit'] = _logit(train_agg_k['pred'])
            glm_logit = sm.GLM(train_agg_k['obs'], sm.add_constant(train_agg_k['pred_logit']),
                               family=sm.families.Binomial()).fit()
            df_list.loc[mask_val, 'response_rescale_logit_glm'] = glm_logit.predict(sm.add_constant(logit_k[mask_val]))
            df_list.loc[mask_avg, 'response_rescale_logit_glm'] += glm_logit.predict(sm.add_constant(logit_k[mask_avg]))

    # Average the "always in-sample" rows (validation == 2 across all folds)
    last_k = n_splits - 1
    mask_avg = df_list[f'validation_{last_k}'] == 2
    df_list.loc[mask_avg, 'response'] = df_list.loc[mask_avg, 'response'] / n_splits
    for col in RESCALE_COLS:
        df_list.loc[mask_avg, col] = df_list.loc[mask_avg, col] / n_splits

    # Map question_id -> variable name
    keep_cols = ['yearid_id', 'question_id', 'response'] + RESCALE_COLS
    long_form = df_list[keep_cols].copy()
    long_form = long_form.merge(
        df_analysis[['question_id', 'variable']].drop_duplicates(), on='question_id', how='left'
    )
    long_form = long_form[['yearid_id', 'variable', 'response'] + RESCALE_COLS]
    del df_list, df_list_reg
    gc.collect()

    # Pivot to wide format
    print('Pivoting to wide format...')
    pivot = long_form.pivot(index='yearid_id', columns='variable')
    wide_form = pivot['response'].reset_index()
    df_hori = wide_form.copy()

    # Merge year info
    df_hori = df_hori.merge(
        df_analysis[['yearid_id', 'yearid', 'year']].drop_duplicates(), on='yearid_id', how='left'
    )

    # Add all rescaled predictions
    for col in RESCALE_COLS:
        suffix = col.replace('response', '')  # e.g. _rescale, _rescale_logit, ...
        rescale_form = pivot[col].add_suffix(suffix).reset_index()
        df_hori = df_hori.merge(rescale_form, on='yearid_id', how='left')
    del pivot
    gc.collect()

    # Add predicted binary (threshold 0.5)
    pred_cols = [c for c in wide_form.columns if c != 'yearid_id']
    bin_df = (wide_form[pred_cols] >= 0.5).astype(int)
    bin_df.columns = [f'{c}_bin0.5' for c in pred_cols]
    bin_df['yearid_id'] = wide_form['yearid_id']
    df_hori = df_hori.merge(bin_df, on='yearid_id', how='left')

    # Add observed binarized values
    obs_pivot = df_analysis.pivot_table(index='yearid_id', columns='variable', values='binarized')
    obs_pivot = obs_pivot.add_suffix('_obs_bin').reset_index()
    df_hori = df_hori.merge(obs_pivot, on='yearid_id', how='left')
    del obs_pivot
    gc.collect()

    # Merge sampling weights
    gss_df = pd.read_parquet(DATA_DIR / 'gss_df_merged.parquet')
    yearid_dict = dict(zip(df_analysis.yearid_id, df_analysis.yearid))
    year_dict = dict(zip(df_analysis.yearid_id, df_analysis.year))
    df_hori['yearid'] = df_hori['yearid_id'].map(yearid_dict)
    df_hori['year'] = df_hori['yearid_id'].map(year_dict)
    df_hori['id'] = df_hori['yearid'] % 10000
    df_hori = df_hori.merge(
        gss_df[['year', 'id', 'wtssall', 'wtssnr', 'wtssnrps']].drop_duplicates(),
        on=['year', 'id'], how='left'
    )

    # Save
    if use_demo is False and use_poli is True:
        out_name = f'{safe_name}_{split_type}_{n_splits}_{batch_size}_{dim}__resample{resample}_wide.parquet'
    else:
        out_name = f'{safe_name}_{split_type}_{n_splits}_{batch_size}_{dim}__resample{resample}_{use_demo}_{use_poli}_wide.parquet'
    out_path = PREDICTIONS_DIR / out_name
    df_hori.to_parquet(out_path)
    print(f'Saved wide parquet -> {out_path} (shape: {df_hori.shape})')


if __name__ == '__main__':
    fire.Fire()
