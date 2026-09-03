#!/usr/bin/env python
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_val_var_year_pairs import load_data_partial  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
PRED_DIR = DATA_DIR / 'predictions'
OUT_DIR  = DATA_DIR / 'fig_table_gen'
N_FOLDS = 10
MODEL_PREFIX = 'maicomputer_alpaca-native'

BIN_ORDER = ['1', '2-3', '4-5', '6-7', '8+']
MODE_ORDER = ['retrodiction', 'interpolation', 'forecast']

log = logging.getLogger('prep_retro_distance_metrics')


def _setup_logging():
    logs_dir = BASE_DIR / 'logs'
    logs_dir.mkdir(exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_file = logs_dir / f'{ts}_prep_retro_distance_metrics.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    log.info(f'Log -> {log_file}')


def classify(y_test, obs_years):
    y = int(y_test)
    others = [int(z) for z in obs_years if int(z) != y]
    if len(others) == 0:
        return 'unknown'
    lo, hi = min(others), max(others)
    if y < lo:
        return 'retrodiction'
    if y > hi:
        return 'forecast'
    return 'interpolation'


def distance_to_nearest(y, train_years):
    if len(train_years) == 0:
        return np.nan
    return float(np.min(np.abs(np.asarray(train_years, dtype=int) - int(y))))


def bin_distance(d):
    if pd.isna(d):
        return np.nan
    d = int(d)
    if d <= 1:
        return '1'
    if d <= 3:
        return '2-3'
    if d <= 5:
        return '4-5'
    if d <= 7:
        return '6-7'
    return '8+'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging()

    log.info('Loading df_analysis_slim ...')
    slim = pd.read_pickle(DATA_DIR / 'df_analysis_slim.pkl')
    slim = slim.loc[slim['binarized'].notna()]
    qid_var = (slim[['question_id', 'variable']].drop_duplicates()
               .set_index('question_id')['variable'].to_dict())
    qid_year = slim[['yearid_id', 'year']].drop_duplicates(
        subset=['yearid_id']).set_index('yearid_id')['year'].to_dict()
    gt = (slim[['yearid_id', 'question_id', 'binarized']]
          .drop_duplicates(['yearid_id', 'question_id']))

    obs_years_by_var = (
        slim[['variable', 'year']].drop_duplicates()
        .groupby('variable')['year']
        .apply(lambda s: np.sort(s.astype(int).unique()))
        .to_dict()
    )
    log.info(f'variables with obs years: {len(obs_years_by_var)}')

    parts = []
    for k in range(N_FOLDS):
        path = PRED_DIR / (
            f'{MODEL_PREFIX}_partial_{k}_{N_FOLDS}_128_50'
            f'__resample1_long.parquet')
        if not path.exists():
            log.warning(f'fold {k} missing: {path.name}')
            continue
        log.info(f'fold {k}: loading predictions + train years ...')
        lp = pd.read_parquet(path, columns=['yearid_id', 'question_id',
                                            f'validation_{k}', f'response_{k}'])
        lp = lp.loc[lp[f'validation_{k}'] == 1,
                    ['yearid_id', 'question_id', f'response_{k}']]
        lp = lp.rename(columns={f'response_{k}': 'pred'})
        lp = lp.merge(gt, on=['yearid_id', 'question_id'], how='inner')
        lp['variable'] = lp['question_id'].map(qid_var)
        lp['year'] = lp['yearid_id'].map(qid_year).astype('Int64')
        lp = lp.dropna(subset=['binarized', 'pred', 'variable', 'year'])
        lp['fold'] = k

        # fold-specific training years per variable
        train, _ = load_data_partial(k=k, n_splits=N_FOLDS, resample=1)
        train_years_by_var = (
            train[['variable', 'year']]
            .drop_duplicates()
            .groupby('variable')['year']
            .apply(lambda s: np.sort(s.astype(int).unique()))
            .to_dict()
        )
        lp['distance_to_nearest'] = [
            distance_to_nearest(y, train_years_by_var.get(v, []))
            for v, y in zip(lp['variable'], lp['year'])
        ]
        log.info(f'  fold {k}: rows={len(lp):,}, '
                 f'missing dist={int(lp["distance_to_nearest"].isna().sum())}')
        parts.append(lp)

    pooled = pd.concat(parts, ignore_index=True)
    pooled['year'] = pooled['year'].astype(int)
    pooled['mode'] = [
        classify(y, obs_years_by_var.get(v, []))
        for v, y in zip(pooled['variable'], pooled['year'])
    ]
    pooled['dist_bin'] = pooled['distance_to_nearest'].map(bin_distance)
    log.info(f'pooled: {len(pooled):,} cells '
             f'(mode={pooled["mode"].value_counts().to_dict()})')

    rows = []
    for mode in MODE_ORDER:
        for b in BIN_ORDER:
            sub = pooled.loc[(pooled['mode'] == mode)
                             & (pooled['dist_bin'] == b)]
            n = len(sub)
            if n == 0:
                rows.append({'mode': mode, 'dist_bin': b,
                             'n': 0, 'auc': None, 'acc': None, 'f1': None})
                continue
            y = sub['binarized'].to_numpy(dtype=np.int32)
            p = sub['pred'].to_numpy(dtype=np.float64)
            yhat = (p >= 0.5).astype(np.int32)
            auc = (float(roc_auc_score(y, p))
                   if np.unique(y).size == 2 else float('nan'))
            acc = float(accuracy_score(y, yhat))
            f1  = float(f1_score(y, yhat, zero_division=0))
            log.info(f'{mode:>14s}  bin {b:>4s}: '
                     f'n={n:>10,}  AUC={auc:.4f}  '
                     f'Acc={acc:.4f}  F1={f1:.4f}')
            rows.append({'mode': mode, 'dist_bin': b, 'n': n,
                         'auc': auc, 'acc': acc, 'f1': f1})

    out = pd.DataFrame(rows, columns=['mode', 'dist_bin',
                                       'n', 'auc', 'acc', 'f1'])
    out_path = OUT_DIR / 'retro_distance_metrics.csv'
    out.to_csv(out_path, index=False)
    log.info(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
