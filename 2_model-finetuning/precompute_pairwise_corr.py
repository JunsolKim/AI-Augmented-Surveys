#!/usr/bin/env python
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
CACHE = DATA / 'cache'

log = logging.getLogger('precompute_pairwise_corr')


def _setup_logging():
    logs = BASE / 'logs'; logs.mkdir(exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    lf = logs / f'{ts}_precompute_pairwise_corr.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(lf)],
    )
    log.info(f'Log -> {lf}')


def _t(t0, msg):
    log.info(f'[{time.time() - t0:6.1f}s] {msg}')


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    _setup_logging()
    t0 = time.time()

    log.info('Loading df_analysis_slim.pkl ...')
    df = pd.read_pickle(DATA / 'df_analysis_slim.pkl')
    df = df.loc[df['binarized'].notna(),
                ['yearid_id', 'question_id', 'binarized']]
    df = df.drop_duplicates(['yearid_id', 'question_id'])
    n_users = int(df['yearid_id'].max()) + 1
    n_items = int(df['question_id'].max()) + 1
    _t(t0, f'obs={len(df):,}  n_users={n_users}  n_items={n_items}')

    # Dense matrices: X (NaN->0) and M (0/1 mask). float32 to fit memory.
    log.info('Building dense X (zeros for missing) and mask M ...')
    X = np.zeros((n_users, n_items), dtype=np.float32)
    M = np.zeros((n_users, n_items), dtype=np.float32)
    X[df['yearid_id'].values, df['question_id'].values] = df['binarized'].astype(np.float32).values
    M[df['yearid_id'].values, df['question_id'].values] = 1.0
    _t(t0, f'X={X.shape}  nnz={int(M.sum()):,}  dens={M.mean():.3%}')

    # Mask-BLAS correlation (pairwise complete)
    log.info('Computing pairwise-complete corr via mask BLAS (6 GEMMs) ...')
    _t(t0, '  GEMM 1/6: count = M.T @ M')
    count = M.T @ M                      # (p, p) number of complete pairs

    _t(t0, '  GEMM 2/6: Sxy = X.T @ X')
    Sxy = X.T @ X                        # sum of x*y over pairs

    _t(t0, '  GEMM 3/6: Sx = X.T @ M')
    Sx = X.T @ M                         # sum of x over (j,k) paired obs

    # Sy = Sx.T (by symmetry of definition when x=y=same field; general case:
    # Sy(j,k) = sum_i M_ij * M_ik * X_ik = (M.T @ X)[j,k] = X.T @ M transposed)
    _t(t0, '  (Sy = Sx.T  skipped)')
    Sy = Sx.T

    _t(t0, '  GEMM 4/6: Sxx = (X^2).T @ M')
    X2 = X * X
    Sxx = X2.T @ M                       # sum of x^2 over paired obs
    _t(t0, '  (Syy = Sxx.T  skipped)')
    Syy = Sxx.T

    _t(t0, 'Combining into correlation ...')
    count_safe = np.maximum(count, 1.0)
    mean_x = Sx / count_safe
    mean_y = Sy / count_safe
    cov    = Sxy / count_safe - mean_x * mean_y
    var_x  = Sxx / count_safe - mean_x * mean_x
    var_y  = Syy / count_safe - mean_y * mean_y
    # Numerical safety
    var_x = np.clip(var_x, 0.0, None)
    var_y = np.clip(var_y, 0.0, None)
    denom = np.sqrt(var_x * var_y)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(denom > 1e-12, cov / denom, 0.0)
    corr = np.clip(corr, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(corr, 1.0)
    _t(t0, f'corr shape={corr.shape}  mean|r|={np.mean(np.abs(corr)):.3f}  '
           f'min={corr.min():.3f}  max={corr.max():.3f}')

    # Drop rows where count == 0 (pairs never co-observed): set corr to 0
    zero_cnt = (count < 1).sum()
    if zero_cnt:
        log.info(f'  {zero_cnt} (j,k) pairs never co-observed -> corr=0')

    # Save
    corr_path  = CACHE / 'pairwise_corr_full.parquet'
    count_path = CACHE / 'pairwise_corr_count.parquet'
    # Save as a DataFrame with 3572 numeric columns indexed 0..3571.
    cols = [f'q{i}' for i in range(n_items)]
    pd.DataFrame(corr, columns=cols).to_parquet(corr_path, index=False)
    pd.DataFrame(count.astype(np.int32), columns=cols).to_parquet(count_path, index=False)
    _t(t0, f'Saved: {corr_path.name} ({corr_path.stat().st_size/1e6:.1f} MB)')
    _t(t0, f'Saved: {count_path.name} ({count_path.stat().st_size/1e6:.1f} MB)')
    _t(t0, 'done.')


if __name__ == '__main__':
    main()
