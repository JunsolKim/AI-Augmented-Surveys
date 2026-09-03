from pathlib import Path

import fire
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'


def load_data_partial(k, n_splits=10, resample=1, use_poli=True, seed=42):
    """Reproduce the 'partial' branch of utils.load_data without importing TF."""
    slim_path = DATA_DIR / 'df_analysis_slim.pkl'
    src = slim_path if slim_path.exists() else DATA_DIR / 'df_analysis.pkl'
    df = pd.read_pickle(src)
    df = df.drop_duplicates(['yearid_id', 'question_id'])

    unique_q = df[['question_id', 'year_order']].drop_duplicates().reset_index(drop=True)
    unique_q['count'] = unique_q.groupby('question_id')['year_order'].transform('count')
    unique_q_two = pd.DataFrame(unique_q.loc[unique_q['count'] > 1]).reset_index()
    unique_q_one = pd.DataFrame(unique_q.loc[unique_q['count'] == 1]).reset_index()

    kfold = KFold(shuffle=True, n_splits=n_splits, random_state=seed)
    val_q2 = unique_q_two.loc[list(kfold.split(unique_q_two))[k][1]]
    kfold = KFold(shuffle=True, n_splits=n_splits, random_state=seed)
    val_q1 = unique_q_one.loc[list(kfold.split(unique_q_one))[k][1]]

    val_q = pd.concat([val_q1, val_q2], axis=0).drop('count', axis=1)
    val_q['validation'] = 1
    df = df.merge(val_q, on=['question_id', 'year_order'], how='left')
    df['validation'] = df['validation'].fillna(0)

    df = df.loc[df.binarized.notna() & df.yearid_id.notna() & df.question_id.notna()].copy()

    train = (
        df.loc[df.validation == 0]
        .groupby('yearid_id')
        .sample(frac=resample, random_state=seed)
        .reset_index(drop=True)
    )
    if not use_poli:
        poli_vars = ['polviews1', 'polviews2', 'polviews3',
                     'partyid_dem', 'partyid_rep', 'partyid_ind', 'partyid_oth']
        train = train.loc[~train.variable.isin(poli_vars)]
    val = df.loc[df.validation == 1].copy()
    return train, val


def main(n_splits=10, resample=1, use_poli=True, out_csv=None):
    split_type = 'partial'
    if out_csv is None:
        out_csv = DATA_DIR / f'val_var_year_pairs_{split_type}.csv'
    out_csv = Path(out_csv)

    per_fold_rows = []        # (fold, variable, year)
    per_fold_summary = []     # (fold, n_common_vars, n_val_var_year_pairs)

    for k in range(n_splits):
        print(f'\n=== fold {k} ===')
        train, val = load_data_partial(
            k=k, n_splits=n_splits, resample=resample, use_poli=use_poli,
        )

        train_vars = set(train['variable'].unique())
        val_vars = set(val['variable'].unique())
        common_vars = train_vars & val_vars
        print(f'  train vars: {len(train_vars)}, '
              f'val vars: {len(val_vars)}, '
              f'common: {len(common_vars)}')

        val_common = val.loc[val['variable'].isin(common_vars),
                             ['variable', 'year']].drop_duplicates()
        val_common = val_common.sort_values(['variable', 'year']).reset_index(drop=True)
        val_common.insert(0, 'fold', k)
        per_fold_rows.append(val_common)

        per_fold_summary.append({
            'fold': k,
            'n_train_vars': len(train_vars),
            'n_val_vars': len(val_vars),
            'n_common_vars': len(common_vars),
            'n_val_var_year_pairs': len(val_common),
        })

    all_pairs = pd.concat(per_fold_rows, axis=0, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_pairs.to_csv(out_csv, index=False)

    summary = pd.DataFrame(per_fold_summary)
    print('\nPer-fold summary:')
    print(summary.to_string(index=False))
    print(f'\nWrote {len(all_pairs):,} (fold, variable, year) rows -> {out_csv}')


if __name__ == '__main__':
    fire.Fire(main)
