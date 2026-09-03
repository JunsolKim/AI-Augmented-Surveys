import os
import random
import time

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _seed_all(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def load_data(split_type='partial', k=0, n_splits=10, resample=1, use_demo=False, use_poli=True,
              exclude_vars=None, max_q=0):
    """Load df_analysis and create train/val split.

    Args:
        split_type: 'impute' | 'partial' | 'total' | 'module' | 'panel'
                    | 'mar' | 'mnar'
        k: fold index (0 to n_splits-1)
        n_splits: number of CV folds
        resample: fraction of training data to keep (1 = all)
        use_demo: include demographic questions as training features
        use_poli: include political variables (polviews, partyid)

    Returns:
        (df_analysis, train, val) DataFrames
    """
    _seed_all()
    tqdm.pandas()

    start_time = time.time()
    slim_path = DATA_DIR / 'df_analysis_slim.pkl'
    src = slim_path if slim_path.exists() else DATA_DIR / 'df_analysis.pkl'
    df_analysis = pd.read_pickle(src)
    print(f'Loaded {src.name} in {time.time() - start_time:.1f}s')
    df_analysis = df_analysis.drop_duplicates(['yearid_id', 'question_id'])
    print(f'Split type: {split_type}')

    if split_type == 'total':
        # Hold out entire question_ids (all years)
        unique_q = df_analysis[['question_id']].drop_duplicates().reset_index(drop=True)
        unique_q['count'] = unique_q.groupby('question_id')['question_id'].transform('count')
        kfold = KFold(shuffle=True, n_splits=n_splits, random_state=42)
        folded = list(kfold.split(unique_q))[k]
        val_q = unique_q.loc[folded[1]]
        val_q = val_q.drop('count', axis=1)
        val_q['validation'] = 1
        df_analysis = df_analysis.merge(val_q, on=['question_id'], how='left')
        df_analysis['validation'] = df_analysis['validation'].fillna(0)

    elif split_type == 'partial':
        # Hold out (question_id, year_order) pairs
        unique_q = df_analysis[['question_id', 'year_order']].drop_duplicates().reset_index(drop=True)
        unique_q['count'] = unique_q.groupby('question_id')['year_order'].transform('count')
        unique_q_two = pd.DataFrame(unique_q.loc[unique_q['count'] > 1]).reset_index()
        unique_q_one = pd.DataFrame(unique_q.loc[unique_q['count'] == 1]).reset_index()
        kfold = KFold(shuffle=True, n_splits=n_splits, random_state=42)
        folded = list(kfold.split(unique_q_two))[k]
        val_q2 = unique_q_two.loc[folded[1]]
        kfold = KFold(shuffle=True, n_splits=n_splits, random_state=42)
        folded = list(kfold.split(unique_q_one))[k]
        val_q1 = unique_q_one.loc[folded[1]]
        val_q = pd.concat([val_q1, val_q2], axis=0)
        val_q = val_q.drop('count', axis=1)
        val_q['validation'] = 1
        df_analysis = df_analysis.merge(val_q, on=['question_id', 'year_order'], how='left')
        df_analysis['validation'] = df_analysis['validation'].fillna(0)

    elif split_type == 'impute':
        # Hold out individual (question_id, yearid_id) observations
        unique_q = df_analysis[['question_id', 'yearid_id']].drop_duplicates().reset_index(drop=True)
        unique_q['count'] = unique_q.groupby('question_id')['yearid_id'].transform('count')
        unique_q_two = pd.DataFrame(unique_q.loc[unique_q['count'] > 1]).reset_index()
        kfold = KFold(shuffle=True, n_splits=n_splits, random_state=42)
        folded = list(kfold.split(unique_q_two))[k]
        val_q = unique_q_two.loc[folded[1]]
        val_q = val_q.drop('count', axis=1)
        val_q['validation'] = 1
        df_analysis = df_analysis.merge(val_q, on=['question_id', 'yearid_id'], how='left')
        df_analysis['validation'] = df_analysis['validation'].fillna(0)

    elif split_type == 'module':
        # Hold out (module, year_order) pairs — Core module always in training
        unique_q = df_analysis.loc[df_analysis.module != 'Core',
                                   ['module', 'year_order']].drop_duplicates().reset_index(drop=True)
        unique_q['count'] = unique_q.groupby('module')['year_order'].transform('count')
        unique_q_two = pd.DataFrame(unique_q.loc[unique_q['count'] > 1]).reset_index()
        unique_q_one = pd.DataFrame(unique_q.loc[unique_q['count'] == 1]).reset_index()
        kfold = KFold(shuffle=True, n_splits=n_splits, random_state=42)
        folded = list(kfold.split(unique_q_two))[k]
        val_q2 = unique_q_two.loc[folded[1]]
        kfold = KFold(shuffle=True, n_splits=n_splits, random_state=42)
        folded = list(kfold.split(unique_q_one))[k]
        val_q1 = unique_q_one.loc[folded[1]]
        val_q = pd.concat([val_q1, val_q2], axis=0)
        val_q = val_q.drop('count', axis=1)
        val_q['validation'] = 1
        df_analysis = df_analysis.merge(val_q, on=['module', 'year_order'], how='left')
        df_analysis['validation'] = df_analysis['validation'].fillna(0)

    elif split_type == 'panel':
        df_analysis = pd.read_pickle(DATA_DIR / 'df_analysis_panel.pkl')
        unique_q = df_analysis[['yearid_id']].drop_duplicates().reset_index(drop=True)
        unique_q['count'] = unique_q.groupby('yearid_id')['yearid_id'].transform('count')
        kfold = KFold(shuffle=True, n_splits=n_splits, random_state=42)
        folded = list(kfold.split(unique_q))[k]
        val_q = unique_q.loc[folded[1]]
        val_q = val_q.drop('count', axis=1)
        val_q['validation'] = 1
        df_analysis = df_analysis.merge(val_q, on=['yearid_id'], how='left')
        df_analysis['validation'] = df_analysis['validation'].fillna(0)

    elif split_type in ('mar', 'mnar'):
        # Load pre-computed deterministic missing mask from step6b.
        # validation=1 for top 10% by predicted missing probability per (variable, year).
        mask_path = DATA_DIR / f'missing_mask_{split_type}.parquet'
        mask = pd.read_parquet(mask_path)
        df_analysis = df_analysis.merge(
            mask[['yearid_id', 'question_id', 'validation']],
            on=['yearid_id', 'question_id'], how='left',
        )
        df_analysis['validation'] = df_analysis['validation'].fillna(0)

    else:
        raise ValueError(f'Unknown split_type: {split_type}')

    # Optionally append demographic questions
    if use_demo:
        demo_long_path = DATA_DIR / 'demographics_long.csv'
        demographics_long = pd.read_csv(demo_long_path)
        demographics_long['question_id'] = demographics_long.question_id + df_analysis.question_id.max() + 1
        demographics_long['validation'] = 0
        df_analysis = pd.concat([df_analysis, demographics_long], axis=0, ignore_index=True)

    # Filter valid rows
    df_analysis = df_analysis.loc[
        df_analysis.binarized.notna() & df_analysis.yearid_id.notna() & df_analysis.question_id.notna()
    ].copy()

    # Train/val split
    train = df_analysis.loc[df_analysis.validation == 0].groupby('yearid_id').sample(
        frac=resample, random_state=42
    ).reset_index(drop=True)
    if not use_poli:
        poli_vars = ['polviews1', 'polviews2', 'polviews3', 'partyid_dem', 'partyid_rep', 'partyid_ind', 'partyid_oth']
        train = train.loc[~train.variable.isin(poli_vars)]
    if exclude_vars:
        train = train.loc[~train.variable.isin(exclude_vars)]

    # Limit questions per respondent for years 2016/2018/2021 (figure A15)
    # Only drops from train; validation stays as the original impute 10%.
    if max_q > 0:
        nq_years = {2016, 2018, 2021}
        rng = np.random.RandomState(42)
        keep_idx = []
        for yid, grp in train.groupby('yearid_id'):
            if 'year' in grp.columns and grp['year'].iloc[0] in nq_years and len(grp) > max_q:
                keep_idx.append(grp.sample(n=max_q, random_state=rng).index)
            else:
                keep_idx.append(grp.index)
        train = train.loc[pd.Index(np.concatenate(keep_idx))]
        print(f'  max_q={max_q}: limited to {max_q} questions/person for 2016/2018/2021')

    val = df_analysis.loc[df_analysis.validation == 1].copy()

    print(f'Train: {len(train)}, Val: {len(val)}')
    return df_analysis, train, val
