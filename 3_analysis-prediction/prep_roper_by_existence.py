#!/usr/bin/env python
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
GEMMA_DIR = DATA_DIR / 'roper_prep' / 'results' / 'colab_llm_gemma'
META = DATA_DIR / 'roper_prep' / 'roper_question_meta.parquet'
POP_MEAN = DATA_DIR / 'processed' / 'figure4_pop_mean.parquet'
VAL_PAIRS = DATA_DIR / 'val_var_year_pairs_partial.csv'
OUT = DATA_DIR / 'fig_table_gen' / 'roper_by_existence.parquet'

COS_MIN = 0.85
CONF_MIN = 0.85

log = logging.getLogger('prep_roper_by_existence')


def _setup_logging():
    logs_dir = BASE_DIR / 'logs'
    logs_dir.mkdir(exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    fh = logs_dir / f'{ts}_prep_roper_by_existence.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(fh)],
    )
    log.info(f'Log -> {fh}')


def main():
    _setup_logging()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if not META.exists():
        sys.exit(f'meta cache missing: {META} — run prep_roper_question_meta.py first')

    log.info('Loading gemma binarize output (+ fill) ...')
    cols = ['gss_variable', 'year', 'roper_question', 'roper_question_id',
            'cos_sim', 'roper_yes_pct']
    bo = pd.read_csv(GEMMA_DIR / 'binarize_output.csv', usecols=cols)
    bof = pd.read_csv(GEMMA_DIR / 'binarize_output_fill.csv', usecols=cols)
    bin_df = pd.concat([bo, bof], ignore_index=True)
    bin_df = bin_df.drop_duplicates(['gss_variable', 'roper_question_id', 'year'])
    log.info(f'  binarize rows: {len(bin_df):,}')

    log.info('Loading gemma verify output (Yes pairs only) ...')
    vo = pd.read_csv(
        GEMMA_DIR / 'verify_output.csv',
        usecols=['gss_variable', 'roper_question', 'llm_same', 'llm_confidence'],
    )
    vo = vo.loc[vo['llm_same'] == 'Yes',
                ['gss_variable', 'roper_question', 'llm_confidence']]
    vo = vo.drop_duplicates(['gss_variable', 'roper_question'])
    log.info(f'  verify Yes rows: {len(vo):,}')

    log.info('Loading Roper question meta cache ...')
    meta = pd.read_parquet(META)[['roper_question_id', 'national_adult', 'gss']]

    log.info('Joining: binarize + verify + meta ...')
    df = bin_df.merge(vo, on=['gss_variable', 'roper_question'], how='inner')
    df = df.merge(meta, on='roper_question_id', how='left')
    df['national_adult'] = df['national_adult'].fillna(0).astype(int)
    df['gss'] = df['gss'].fillna(0).astype(int)
    log.info(f'  joined rows: {len(df):,}, '
             f'national_adult==1: {int((df.national_adult == 1).sum()):,}, '
             f'gss==1: {int((df.gss == 1).sum()):,}')

    log.info(f'Filtering: cos_sim >= {COS_MIN}, llm_confidence >= {CONF_MIN}, '
             f'national_adult == 1, gss == 0, roper_yes_pct in [0, 100] ...')
    df['roper_mean'] = pd.to_numeric(df['roper_yes_pct'], errors='coerce') / 100.0
    mask = (
        (df['cos_sim'] >= COS_MIN)
        & (df['llm_confidence'] >= CONF_MIN)
        & (df['national_adult'] == 1)
        & (df['gss'] == 0)
        & df['roper_mean'].between(0.0, 1.0)
    )
    df = df.loc[mask].copy()
    log.info(f'  after filters: {len(df):,} poll-rows')

    log.info('Aggregating per (gss_variable, year) ...')
    df['year'] = df['year'].astype(int)
    before_year = len(df)
    df = df.loc[df['year'] <= 2021].copy()
    log.info(f'  year <= 2021: {len(df):,} (dropped {before_year - len(df):,})')
    agg = (
        df.groupby(['gss_variable', 'year'], as_index=False)
        .agg(
            roper_mean=('roper_mean', 'mean'),
            cos_sim=('cos_sim', 'max'),
            llm_confidence=('llm_confidence', 'max'),
            n_polls=('roper_mean', 'size'),
        )
    )
    agg['national_adult'] = 1
    log.info(f'  aggregated (var, year) cells: {len(agg):,}')

    log.info('Loading calibrated GSS predictions (figure4_pop_mean partial) ...')
    pm = pd.read_parquet(POP_MEAN)
    pm = pm.loc[pm['task'] == 'partial', ['variable', 'year', 'mean', 'pred_type']]
    pm['year'] = pm['year'].astype(int)
    alpaca = (pm.loc[pm['pred_type'] == 'rescale_logit_glm',
                     ['variable', 'year', 'mean']]
              .rename(columns={'mean': 'alpaca_mean'}))
    obs = (pm.loc[pm['pred_type'] == 'obs_bin',
                  ['variable', 'year', 'mean']]
           .rename(columns={'mean': 'gss_obs'}))

    log.info('Joining with calibrated GSS predictions ...')
    out = agg.merge(alpaca, left_on=['gss_variable', 'year'],
                    right_on=['variable', 'year'], how='inner').drop(columns=['variable'])
    out = out.merge(obs, left_on=['gss_variable', 'year'],
                    right_on=['variable', 'year'], how='left').drop(columns=['variable'])
    log.info(f'  after pop_mean join: {len(out):,}')

    log.info('Restricting to variables in GSS partial-fold val set ...')
    vp = pd.read_csv(VAL_PAIRS)
    vp['year'] = vp['year'].astype(int)
    val_vars = set(vp['variable'].unique())
    val_pairs = set(zip(vp['variable'], vp['year']))
    before = len(out)
    out = out.loc[out['gss_variable'].isin(val_vars)].copy()
    log.info(f'  kept (var in val set): {len(out):,} / {before}')

    out['has_gss_obs_same_year'] = [
        (v, y) in val_pairs for v, y in zip(out['gss_variable'], out['year'])
    ]
    out['error'] = out['alpaca_mean'] - out['roper_mean']

    cols = ['gss_variable', 'year', 'alpaca_mean', 'gss_obs', 'roper_mean',
            'error', 'cos_sim', 'llm_confidence', 'national_adult',
            'has_gss_obs_same_year', 'n_polls']
    out = out[cols].reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    log.info(f'Saved {out.shape} -> {OUT}')

    for flag in [True, False]:
        sub = out.loc[out['has_gss_obs_same_year'] == flag]
        if len(sub) == 0:
            continue
        from scipy.stats import spearmanr
        rho, _ = spearmanr(sub['alpaca_mean'], sub['roper_mean'])
        mae = float((sub['alpaca_mean'] - sub['roper_mean']).abs().mean())
        log.info(f'  has_gss_obs_same_year={flag}: n={len(sub):,}  '
                 f'Spearman={rho:.4f}  MAE={mae:.4f}')


if __name__ == '__main__':
    main()
