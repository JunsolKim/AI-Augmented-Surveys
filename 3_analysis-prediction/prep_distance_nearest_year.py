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
RETRO_PQ = OUT_DIR / 'retro_interp_forecast_varyear.parquet'
N_SPLITS = 10

log = logging.getLogger('prep_distance_nearest_year')


def _setup_logging():
    logs_dir = BASE_DIR / 'logs'
    logs_dir.mkdir(exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_file = logs_dir / f'{ts}_prep_distance_nearest_year.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    log.info(f'Log -> {log_file}')


def distance_to_nearest(y_test, train_years):
    if len(train_years) == 0:
        return np.nan
    return float(np.min(np.abs(np.asarray(train_years, dtype=int) - int(y_test))))


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

    retro = pd.read_parquet(RETRO_PQ)
    log.info(f'retro  : {retro.shape}, cols={retro.columns.tolist()}')

    distances = []
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
        rk = retro.loc[retro['fold'] == k, ['fold', 'variable', 'year']].copy()
        rk['distance_to_nearest'] = [
            distance_to_nearest(y, train_years_by_var.get(v, []))
            for v, y in zip(rk['variable'], rk['year'])
        ]
        miss = int(rk['distance_to_nearest'].isna().sum())
        log.info(f'  rows={len(rk)}, missing distance (no train years)={miss}')
        distances.append(rk)

    dist = pd.concat(distances, ignore_index=True)
    log.info(f'total distances: {dist.shape}')

    merged = retro.merge(dist, on=['fold', 'variable', 'year'], how='left')
    merged['abs_error'] = (merged['GLM_predicted'] - merged['obs_wavg']).abs()

    keep = ['fold', 'variable', 'year', 'mode', 'distance_to_nearest',
            'GLM_predicted', 'obs_wavg', 'abs_error', 'n_val', 'sw_val']
    keep = [c for c in keep if c in merged.columns]
    out = merged[keep]

    out_path = OUT_DIR / 'distance_nearest_year.parquet'
    out.to_parquet(out_path, index=False)
    log.info(f'Saved {out.shape} -> {out_path}')

    bin_order = ['1', '2-3', '4-5', '6-7', '8+']
    out = out.assign(dist_bin=out['distance_to_nearest'].map(bin_distance))

    log.info('Per-mode summary:')
    for mode in ['retrodiction', 'interpolation', 'forecast']:
        sub = out.loc[out['mode'] == mode]
        if len(sub) == 0:
            log.info(f'  {mode}: n=0')
            continue
        sub_nn = sub.dropna(subset=['distance_to_nearest', 'abs_error'])
        mean_d = float(sub_nn['distance_to_nearest'].mean())
        mean_ae = float(sub_nn['abs_error'].mean())
        if len(sub_nn) >= 3:
            pooled_rho = float(
                sub_nn[['distance_to_nearest', 'abs_error']]
                .corr('spearman').iloc[0, 1])
        else:
            pooled_rho = float('nan')
        log.info(f'  {mode}: n={len(sub)}, mean_dist={mean_d:.3f}, '
                 f'mean_abs_error={mean_ae:.4f}, '
                 f'pooled_spearman(dist, abs_error)={pooled_rho:.4f}')
        for b in bin_order:
            sb = sub_nn.loc[sub_nn['dist_bin'] == b]
            if len(sb) < 3:
                log.info(f'    bin {b:>4s}: n={len(sb)}, spearman=NA')
                continue
            rho = float(sb[['distance_to_nearest', 'abs_error']]
                        .corr('spearman').iloc[0, 1])
            mae_b = float(sb['abs_error'].mean())
            log.info(f'    bin {b:>4s}: n={len(sb)}, '
                     f'mean_abs_error={mae_b:.4f}, spearman(d,|e|)={rho:.4f}')


if __name__ == '__main__':
    main()
