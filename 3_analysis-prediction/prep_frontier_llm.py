#!/usr/bin/env python
import io
import logging
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUT_DIR  = DATA_DIR / 'fig_table_gen'
FRONTIER_ZIP = DATA_DIR / 'frontier.zip'
SLIM     = DATA_DIR / 'df_analysis_slim.pkl'

WIDE = {
    ('alpaca', 'impute'):  DATA_DIR / 'predictions' / 'maicomputer_alpaca-native_impute_10_128_50__resample1_wide.parquet',
    ('alpaca', 'partial'): DATA_DIR / 'predictions' / 'maicomputer_alpaca-native_partial_10_128_50__resample1_wide.parquet',
    ('alpaca', 'total'):   DATA_DIR / 'predictions' / 'maicomputer_alpaca-native_total_10_128_50__resample1_wide.parquet',
    ('mf',     'impute'):  DATA_DIR / 'predictions' / 'mf_impute_10_128_50__resample1_wide.parquet',
    ('mf',     'partial'): DATA_DIR / 'predictions' / 'mf_partial_10_128_50__resample1_wide.parquet',
}

SCENARIOS = ['impute', 'partial', 'total']

NAME_MAP = {
    'gemini25flashpreview0520': 'gemini-2.5-flash',
    'deepseekr10528qwen38b':    'deepseek-r1-0528',
    'llm':                      'gpt',
}
DROP_MODELS = {'mistralsmall3224binstruct'}

STRATEGY_ORDER = ['zeroshot', 'demographics', 'random_k10',
                  'correlation_topk10', 'ExhaustiveCorrelation_topk10']

FNAME_RE = re.compile(
    r'^(?P<strategy>.+)_context_prompt_2018--(?P<model>.+)_responses\.csv$'
)

log = logging.getLogger('prep_frontier_llm')


def _setup_logging():
    logs_dir = BASE_DIR / 'logs'
    logs_dir.mkdir(exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_file = logs_dir / f'{ts}_prep_frontier_llm.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    log.info(f'Log -> {log_file}')


def parse_strategy(stem):
    if stem.startswith('random_k10'):
        return 'random_k10'
    if stem.startswith('ExhaustiveCorrelation_topk10'):
        return 'ExhaustiveCorrelation_topk10'
    if stem.startswith('correlation_topk10'):
        return 'correlation_topk10'
    if stem.startswith('demographics'):
        return 'demographics'
    if stem.startswith('zeroshot'):
        return 'zeroshot'
    raise ValueError(f'Unknown strategy stem: {stem}')


def frontier_inventory():
    with zipfile.ZipFile(FRONTIER_ZIP) as z:
        names = [n for n in z.namelist() if n.endswith('.csv')]
    rows = []
    for n in names:
        m = FNAME_RE.match(n)
        if not m:
            log.warning(f'Skipping unrecognized filename: {n}')
            continue
        model_raw = m.group('model')
        if model_raw in DROP_MODELS:
            log.info(f'Skipping dropped model: {model_raw}')
            continue
        strategy = parse_strategy(n)
        rows.append({'filename': n, 'strategy': strategy,
                     'model_raw': model_raw,
                     'model': NAME_MAP.get(model_raw, model_raw)})
    inv = pd.DataFrame(rows)
    log.info('Frontier inventory:\n' + inv.to_string(index=False))
    return inv


def yearid_from_user_id(user_id_series):
    s = user_id_series.astype(str)
    return (s.str[:4] + s.str[4:8]).astype(int)


def load_frontier_csv(name):
    with zipfile.ZipFile(FRONTIER_ZIP) as z:
        with z.open(name) as f:
            raw = f.read()
    df = pd.read_csv(
        io.BytesIO(raw),
        usecols=['year', 'user_id', 'test_question_id',
                 'test_answer_option', 'llm_answer'],
    )
    df = df.rename(columns={'test_question_id': 'variable',
                            'test_answer_option': 'gt_binarized'})
    df['yearid'] = yearid_from_user_id(df['user_id'])
    df['gt_binarized']  = pd.to_numeric(df['gt_binarized'], errors='coerce')
    df['llm_answer']    = pd.to_numeric(df['llm_answer'], errors='coerce')
    ok = df['gt_binarized'].isin([0, 1]) & df['llm_answer'].isin([0, 1])
    return df.loc[ok, ['yearid', 'variable', 'gt_binarized', 'llm_answer']]


def load_yearid_map():
    df = pd.read_pickle(SLIM)
    d = df.loc[df['year'] == 2018,
               ['yearid', 'yearid_id']].drop_duplicates()
    return d.set_index('yearid')['yearid_id'].to_dict()


def load_wide_for_vars(parquet_path, variables):
    cols = ['yearid_id'] + [v for v in variables]
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    available = set(pf.schema.names)
    keep = [v for v in cols if v in available]
    dropped = [v for v in cols if v not in available]
    if dropped:
        log.warning(f'{parquet_path.name}: dropped {len(dropped)} missing '
                    f'columns (first 5: {dropped[:5]})')
    return pd.read_parquet(parquet_path, columns=keep)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging()

    inv = frontier_inventory()
    yearid_to_id = load_yearid_map()
    log.info(f'yearid->yearid_id map for 2018: {len(yearid_to_id)} entries')

    all_vars = set()
    frontiers = {}
    for _, row in inv.iterrows():
        fdf = load_frontier_csv(row['filename'])
        fdf['yearid_id'] = fdf['yearid'].map(yearid_to_id)
        fdf = fdf.dropna(subset=['yearid_id']).copy()
        fdf['yearid_id'] = fdf['yearid_id'].astype(int)
        frontiers[(row['strategy'], row['model'])] = fdf
        all_vars.update(fdf['variable'].unique())
        log.info(f"{row['strategy']:32s} {row['model']:18s}: "
                 f"{len(fdf):>7d} rows, "
                 f"{fdf['variable'].nunique():>4d} vars")

    baseline_wide = {}
    for (m, sc), path in WIDE.items():
        log.info(f'Loading {m}/{sc} wide: {path.name}')
        baseline_wide[(m, sc)] = load_wide_for_vars(path, sorted(all_vars))

    yid_2018 = set(yearid_to_id.values())

    baseline_long = {}
    for (m, sc), wide in baseline_wide.items():
        w = wide.loc[wide['yearid_id'].isin(yid_2018)].copy()
        long = w.melt(id_vars=['yearid_id'], var_name='variable',
                      value_name='pred').dropna(subset=['pred'])
        long[f'{m}_bin']  = (long['pred'] >= 0.5).astype(int)
        long[f'{m}_pred'] = long['pred'].astype(float)
        baseline_long[(m, sc)] = long[['yearid_id', 'variable',
                                       f'{m}_bin', f'{m}_pred']]
        log.info(f'  {m}/{sc} 2018 val predictions: {len(long)}')

    df_slim = pd.read_pickle(SLIM)
    gt = df_slim.loc[(df_slim['year'] == 2018) & df_slim['binarized'].notna(),
                     ['yearid_id', 'variable', 'binarized']]
    gt['gt'] = gt['binarized'].astype(int)

    def _safe_auc(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return float('nan')
        return float(roc_auc_score(y_true, y_score))

    records = []
    for scenario in SCENARIOS:
        b_alpaca = baseline_long.get(('alpaca', scenario))
        b_mf     = baseline_long.get(('mf',     scenario))
        if b_alpaca is None:
            log.info(f'[{scenario}] no alpaca predictions, skipping')
            continue
        if b_mf is not None:
            both = b_alpaca.merge(b_mf, on=['yearid_id', 'variable'],
                                  how='inner')
            log.info(f'[{scenario}] alpaca+mf intersection: {len(both)}')
        else:
            both = b_alpaca.copy()
            log.info(f'[{scenario}] alpaca-only (no mf for this scenario): '
                     f'{len(both)}')

        for (strategy, model), fdf in frontiers.items():
            j = fdf.merge(both, on=['yearid_id', 'variable'], how='inner')
            j = j.merge(gt[['yearid_id', 'variable', 'gt']],
                        on=['yearid_id', 'variable'], how='inner')
            if len(j) == 0:
                continue
            n = len(j)
            y_true = j['gt'].to_numpy(dtype=np.int32)
            cells = [
                (model,    j['llm_answer'].to_numpy(dtype=np.int32),
                 j['llm_answer'].to_numpy(dtype=float)),
                ('alpaca', j['alpaca_bin'].to_numpy(dtype=np.int32),
                 j['alpaca_pred'].to_numpy(dtype=float)),
            ]
            if 'mf_bin' in j.columns:
                cells.append(
                    ('mf', j['mf_bin'].to_numpy(dtype=np.int32),
                     j['mf_pred'].to_numpy(dtype=float))
                )
            for mdl, yhat, yscore in cells:
                acc = float((yhat == y_true).mean())
                f1  = float(f1_score(y_true, yhat, zero_division=0))
                auc = _safe_auc(y_true, yscore)
                records.append({
                    'scenario': scenario,
                    'strategy': strategy if mdl == model else 'n/a',
                    'model':    mdl,
                    'n':        n,
                    'accuracy': acc,
                    'f1':       f1,
                    'auc':      auc,
                })

    out = pd.DataFrame(records)
    baseline_rows = (
        out.loc[out['strategy'] == 'n/a']
        .groupby(['scenario', 'model'], as_index=False)
        .agg(n=('n', 'mean'),
             accuracy=('accuracy', 'mean'),
             f1=('f1', 'mean'),
             auc=('auc', 'mean'))
        .assign(strategy='all-intersection')
    )
    out = pd.concat([out.loc[out['strategy'] != 'n/a'], baseline_rows],
                    ignore_index=True)
    out = out[['scenario', 'model', 'strategy', 'n', 'accuracy', 'f1', 'auc']]

    out_path = OUT_DIR / 'frontier_llm_metrics.csv'
    out.to_csv(out_path, index=False)
    log.info(f'Saved {out.shape} -> {out_path}')
    log.info('Rows:\n' + out.sort_values(['scenario', 'model',
                                          'strategy']).to_string(index=False))


if __name__ == '__main__':
    main()
