#!/usr/bin/env python
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_val_var_year_pairs import load_data_partial  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUT_DIR  = DATA_DIR / 'fig_table_gen'
VAR_YR   = DATA_DIR / 'predictions' / (
    'maicomputer_alpaca-native_partial_10_128_50__resample1_varyear.parquet')
VAL_PAIRS = DATA_DIR / 'val_var_year_pairs_partial.csv'
N_SPLITS = 10

log = logging.getLogger('prep_retro_interp_forecast')


def _setup_logging():
    logs_dir = BASE_DIR / 'logs'
    logs_dir.mkdir(exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_file = logs_dir / f'{ts}_prep_retro_interp_forecast.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    log.info(f'Log -> {log_file}')


def classify(y_test, obs_years):
    """Variable-level classification excluding the held-out year itself."""
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging()

    val_pairs = pd.read_csv(VAL_PAIRS)
    log.info(f'val_pairs: {val_pairs.shape}')

    varyear = pd.read_parquet(VAR_YR)
    log.info(f'varyear : {varyear.shape}')

    # Variable-level observation years: union across folds of (fold k train U val).
    # Equivalent to the full set of observation years per variable in the data.
    log.info('Building variable-level observation year map ...')
    df_slim_path = DATA_DIR / 'df_analysis_slim.pkl'
    src = df_slim_path if df_slim_path.exists() else DATA_DIR / 'df_analysis.pkl'
    df_full = pd.read_pickle(src)
    df_full = df_full.loc[df_full['binarized'].notna()]
    obs_years_by_var = (
        df_full[['variable', 'year']].drop_duplicates()
        .groupby('variable')['year']
        .apply(lambda s: np.sort(s.astype(int).unique()))
        .to_dict()
    )
    log.info(f'variables with obs years: {len(obs_years_by_var)}')

    labeled = []
    fold_labels = []  # fold-level labels for the sensitivity log
    for k in range(N_SPLITS):
        log.info(f'--- fold {k} ---')
        train, _ = load_data_partial(k=k, n_splits=N_SPLITS, resample=1)
        train_years_by_var = (
            train[['variable', 'year']]
            .drop_duplicates()
            .groupby('variable')['year']
            .apply(lambda s: np.sort(s.astype(int).unique()))
            .to_dict()
        )
        vk = val_pairs.loc[val_pairs['fold'] == k].copy()
        vk['mode'] = [
            classify(y, obs_years_by_var.get(v, []))
            for v, y in zip(vk['variable'], vk['year'])
        ]
        vk['mode_foldlevel_legacy'] = [
            classify(y, train_years_by_var.get(v, []))
            for v, y in zip(vk['variable'], vk['year'])
        ]
        var_counts = vk['mode'].value_counts().to_dict()
        fold_counts = vk['mode_foldlevel_legacy'].value_counts().to_dict()
        log.info(f'  variable-level: {var_counts}')
        log.info(f'  fold-level    : {fold_counts}')
        labeled.append(vk)
        fold_labels.append(vk[['fold', 'variable', 'year',
                               'mode_foldlevel_legacy']])

    lab = pd.concat(labeled, ignore_index=True)
    total_var = lab['mode'].value_counts().to_dict()
    total_fold = lab['mode_foldlevel_legacy'].value_counts().to_dict()
    log.info(f'TOTAL variable-level: {lab.shape} {total_var}')
    log.info(f'TOTAL fold-level    : {lab.shape} {total_fold}')

    merged = lab.merge(varyear, on=['variable', 'year'], how='left')
    miss = merged['pred_logit_wavg'].isna().sum()
    log.info(f'merged: {merged.shape}, missing pred_logit_wavg: {miss}')
    merged = merged.loc[merged['pred_logit_wavg'].notna()].reset_index(drop=True)

    keep = ['fold', 'variable', 'year', 'mode', 'mode_foldlevel_legacy',
            'pred_logit_wavg', 'GLM_predicted', 'obs_wavg',
            'sw_val', 'n_val']
    keep = [c for c in keep if c in merged.columns]
    out = merged[keep]

    out_path = OUT_DIR / 'retro_interp_forecast_varyear.parquet'
    out.to_parquet(out_path, index=False)
    log.info(f'Saved {out.shape} -> {out_path}')

    log.info('Per-mode (variable-level) summary:')
    for mode in ['retrodiction', 'interpolation', 'forecast']:
        sub = out.loc[out['mode'] == mode]
        if len(sub) == 0:
            log.info(f'  {mode}: n=0')
            continue
        mae_logit = float((sub['GLM_predicted'] - sub['obs_wavg']).abs().mean())
        pear = sub[['GLM_predicted', 'obs_wavg']].corr('pearson').iloc[0, 1]
        spear = sub[['GLM_predicted', 'obs_wavg']].corr('spearman').iloc[0, 1]
        log.info(f'  {mode}: n={len(sub)}, MAE(GLM)={mae_logit:.4f}, '
                 f'pearson={pear:.4f}, spearman={spear:.4f}')


if __name__ == '__main__':
    main()
