#!/usr/bin/env python
import logging
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUT_DIR  = DATA_DIR / 'fig_table_gen'
PRED_DIR = DATA_DIR / 'predictions'
STATA_ZIP = DATA_DIR / 'GSS_stata.zip'
N_MIN = 500
N_FOLDS = 10

log = logging.getLogger('prep_polviews_rank')


def _setup_logging():
    logs_dir = BASE_DIR / 'logs'
    logs_dir.mkdir(exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_file = logs_dir / f'{ts}_prep_polviews_rank.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    log.info(f'Log -> {log_file}')


POLVIEWS_ORDER = [
    'extremely liberal', 'liberal', 'slightly liberal',
    'moderate, middle of the road',
    'slightly conservative', 'conservative', 'extremely conservative',
]


def load_polviews_ordinal():
    """Return DataFrame with cols: year, id, polviews_ord (1..7)."""
    log.info(f'Loading polviews from {STATA_ZIP.name}')
    with zipfile.ZipFile(STATA_ZIP) as z:
        with z.open('gss7224_r3.dta') as f:
            df = pd.read_stata(f, columns=['year', 'id', 'polviews'])
    df['year'] = df['year'].astype(int)
    df['id']   = df['id'].astype(int)
    mapping = {name: i + 1 for i, name in enumerate(POLVIEWS_ORDER)}
    df['polviews_ord'] = df['polviews'].astype(object).map(mapping)
    log.info(f'polviews rows: {len(df)}  non-null: {int(df["polviews_ord"].notna().sum())}')
    return df[['year', 'id', 'polviews_ord']]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging()

    pv = load_polviews_ordinal().dropna(subset=['polviews_ord'])

    log.info('Loading df_analysis_slim.pkl ...')
    df_slim = pd.read_pickle(DATA_DIR / 'df_analysis_slim.pkl')
    df_slim = df_slim.loc[df_slim['binarized'].notna(),
                          ['yearid_id', 'yearid', 'variable',
                           'binarized', 'year']]
    df_slim['id'] = (df_slim['yearid'].astype(int) % 10000).astype(int)
    df_slim['year'] = df_slim['year'].astype(int)

    merged = df_slim.merge(pv, on=['year', 'id'], how='inner')
    log.info(f'merged cells with polviews: {len(merged)}')

    drop_vars = [v for v in merged['variable'].unique()
                 if str(v).startswith('polviews')]
    log.info(f'Dropping polviews-derived vars: {drop_vars}')
    merged = merged.loc[~merged['variable'].isin(drop_vars)]

    log.info('Computing per-variable Spearman correlations ...')
    records = []
    for var, g in merged.groupby('variable', sort=False):
        if len(g) < N_MIN:
            continue
        x = g['polviews_ord'].to_numpy(dtype=np.int8)
        y = g['binarized'].to_numpy(dtype=np.float32)
        if np.unique(y).size < 2:
            continue
        rho, _ = spearmanr(x, y)
        records.append({'variable': var, 'n_resp': int(len(g)),
                        'spearman_rho': float(rho)})

    out = pd.DataFrame(records)
    out['abs_rho'] = out['spearman_rho'].abs()

    vq = pickle.load(open(DATA_DIR / 'var_question_dict.pkl', 'rb'))
    vd = pickle.load(open(DATA_DIR / 'var_description_dict.pkl', 'rb'))
    out['question'] = out['variable'].map(
        lambda v: ' '.join(str(vq.get(v, '')).split()))
    out['description'] = out['variable'].map(
        lambda v: ' '.join(str(vd.get(v, '')).split()))

    log.info('Computing per-variable Alpaca partial AUC ...')
    big = pd.read_pickle(DATA_DIR / 'df_analysis_slim.pkl')
    qid_to_var = (big[['question_id', 'variable']].drop_duplicates()
                  .set_index('question_id')['variable'].to_dict())

    auc_rows = []
    for k in range(N_FOLDS):
        path = PRED_DIR / (
            f'maicomputer_alpaca-native_partial_{k}_{N_FOLDS}_128_50'
            f'__resample1_long.parquet')
        lp = pd.read_parquet(path)
        val_col  = f'validation_{k}'
        resp_col = f'response_{k}'
        lp = lp.loc[lp[val_col] == 1, ['yearid_id', 'question_id', resp_col]]
        lp = lp.merge(big[['yearid_id', 'question_id', 'binarized']]
                      .dropna(subset=['binarized']),
                      on=['yearid_id', 'question_id'], how='inner')
        lp = lp.rename(columns={resp_col: 'pred'})
        auc_rows.append(lp[['question_id', 'pred', 'binarized']])
        log.info(f'  fold {k} val rows w/ gt: {len(lp)}')
    alp = pd.concat(auc_rows, ignore_index=True)
    alp['variable'] = alp['question_id'].map(qid_to_var)

    auc_records = []
    for var, g in alp.groupby('variable', sort=False):
        y = g['binarized'].to_numpy(dtype=np.int32)
        if np.unique(y).size < 2 or len(g) < 20:
            auc = np.nan
        else:
            auc = roc_auc_score(y, g['pred'].to_numpy(dtype=np.float32))
        auc_records.append({'variable': var,
                            'alpaca_partial_auc': auc,
                            'alpaca_partial_n': int(len(g))})
    alp_var = pd.DataFrame(auc_records)
    log.info(f'Alpaca per-variable AUC rows: {len(alp_var)}')

    out = out.merge(alp_var, on='variable', how='left')

    out = out[['variable', 'description', 'question', 'n_resp',
               'spearman_rho', 'abs_rho',
               'alpaca_partial_auc', 'alpaca_partial_n']]
    out = out.sort_values('abs_rho', ascending=False).reset_index(drop=True)

    out_path = OUT_DIR / 'polviews_rank.parquet'
    out.to_parquet(out_path, index=False)
    log.info(f'Saved {out.shape} -> {out_path}')

    log.info('Top 20 by |rho|:')
    log.info('\n' + out.head(20).to_string(index=False, max_colwidth=60))


if __name__ == '__main__':
    main()
