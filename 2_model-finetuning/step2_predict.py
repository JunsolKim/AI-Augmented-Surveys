import gc
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # required for tensorflow_recommenders (Keras 2)
import pickle
import random
from glob import glob
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf

from utils import load_data
from models import build_dcn_model

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "data" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"


def _weight_name(model_name, split_type, k, n_splits, batch_size, dim, resample, use_demo, use_poli, n_exclude=0, max_q=0):
    safe = model_name.replace('/', '_')
    base = f'{safe}_{split_type}_k{k}_{n_splits}_{batch_size}_{dim}__resample{resample}'
    if n_exclude > 0:
        base = f'{base}_exclude{n_exclude}'
    if max_q > 0:
        base = f'{base}_nq{max_q}'
    if use_demo is False and use_poli is True:
        return base
    return f'{base}_{use_demo}_{use_poli}'


def _find_best_weights(wname):
    """Find the best checkpoint weights file."""
    candidates = sorted(glob(str(MODELS_DIR / f'{wname}_best*')))
    if candidates:
        path = candidates[-1].replace('.index', '')
        return path
    # Fallback: final weights
    candidates = sorted(glob(str(MODELS_DIR / f'{wname}_final*')))
    if candidates:
        return candidates[-1].replace('.index', '')
    raise FileNotFoundError(f'No weights found for {wname} in {MODELS_DIR}')


def dcn_fold(model_name='chavinlo/alpaca-native', split_type='partial', k=0, n_splits=10,
             resample=1, batch_size=128, dim=50, use_demo=False, use_poli=True,
             n_exclude=0, max_q=0, val_only=False):
    """Generate predictions for one fold using trained DCN model."""
    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    df_analysis, train, val = load_data(
        split_type=split_type, k=k, n_splits=n_splits,
        resample=resample, use_demo=use_demo, use_poli=use_poli,
    )

    # Build prediction grid: all (yearid_id, question_id, year_order) combos
    df_test = df_analysis[['yearid_id', 'year_order']].drop_duplicates().sort_values(['yearid_id', 'year_order'])
    df_list = []
    for q in range(int(df_analysis['question_id'].max()) + 1):
        temp = df_test.copy()
        temp['question_id'] = q
        df_list.append(temp)
    df_list = pd.concat(df_list, axis=0).reset_index(drop=True)

    # Load model and predict
    wname = _weight_name(model_name, split_type, k, n_splits, batch_size, dim, resample, use_demo, use_poli, n_exclude=n_exclude, max_q=max_q)
    weight_path = _find_best_weights(wname)
    print(f'Loading weights: {weight_path}')

    model = build_dcn_model(model_name, df_analysis, dim=dim)
    model.load_weights(weight_path)

    pred_features = {
        'individual_id': np.array(df_list['yearid_id']),
        'question_id': np.array(df_list['question_id']),
        'year_id': np.array(df_list['year_order']),
    }
    df_list[f'response_{k}'] = model.predict(pred_features, verbose=1, batch_size=4096)

    # Mark validation rows
    if split_type in ('impute', 'mar', 'mnar'):
        train_q = train[['question_id', 'yearid_id']].drop_duplicates()
        train_q[f'validation_{k}'] = 0
        val_q = val[['question_id', 'yearid_id']].drop_duplicates()
        val_q[f'validation_{k}'] = 1
        train_val_q = pd.concat([train_q, val_q], axis=0)
        df_list = df_list.merge(train_val_q, on=['question_id', 'yearid_id'], how='left')
    else:
        train_q = train[['question_id', 'year_order']].drop_duplicates()
        train_q[f'validation_{k}'] = 0
        val_q = val[['question_id', 'year_order']].drop_duplicates()
        val_q[f'validation_{k}'] = 1
        train_val_q = pd.concat([train_q, val_q], axis=0)
        df_list = df_list.merge(train_val_q, on=['question_id', 'year_order'], how='left')
    df_list[f'validation_{k}'] = df_list[f'validation_{k}'].fillna(2)

    # Keep only train/validation rows if requested (drop validation==2)
    if val_only:
        df_list = df_list[df_list[f'validation_{k}'] != 2].reset_index(drop=True)
        print(f'val_only: kept {len(df_list)} rows (train + validation)')

    # Save
    safe_name = model_name.replace('/', '_')
    base_out = f'{safe_name}_{split_type}_{k}_{n_splits}_{batch_size}_{dim}__resample{resample}'
    if n_exclude > 0:
        base_out = f'{base_out}_exclude{n_exclude}'
    if max_q > 0:
        base_out = f'{base_out}_nq{max_q}'
    if use_demo is False and use_poli is True:
        out_name = f'{base_out}_long.parquet'
    else:
        out_name = f'{base_out}_{use_demo}_{use_poli}_long.parquet'
    out_path = PREDICTIONS_DIR / out_name
    df_list.to_parquet(out_path)
    print(f'Saved -> {out_path} ({len(df_list)} rows)')

    del model, df_list
    tf.keras.backend.clear_session()
    gc.collect()


def mf_fold(split_type='partial', k=0, n_splits=10, resample=1, batch_size=128, dim=50,
            use_demo=False, use_poli=True, val_only=False):
    """Generate predictions for one fold using trained MF model."""
    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    np.random.seed(42)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    df_analysis, train, val = load_data(
        split_type=split_type, k=k, n_splits=n_splits,
        resample=resample, use_demo=use_demo, use_poli=use_poli,
    )

    # Load MF model
    mf_path = MODELS_DIR / f'mf_{split_type}_k{k}_{n_splits}_{dim}__resample{resample}.pkl'
    model_mf = pickle.load(open(mf_path, 'rb'))

    # Build prediction grid
    df_test = df_analysis[['yearid_id', 'year_order']].drop_duplicates().sort_values(['yearid_id', 'year_order'])
    df_list = []
    for q in range(int(df_analysis['question_id'].max()) + 1):
        temp = df_test.copy()
        temp['question_id'] = q
        df_list.append(temp)
    df_list = pd.concat(df_list, axis=0).reset_index(drop=True)

    pred_matrix = model_mf.user_factors.dot(model_mf.item_factors.T)
    df_list[f'response_{k}'] = pred_matrix[
        np.array(df_list['yearid_id']), np.array(df_list['question_id'])
    ]

    # Mark validation
    if split_type in ('impute', 'mar', 'mnar'):
        merge_on = ['question_id', 'yearid_id']
    else:
        merge_on = ['question_id', 'year_order']
    train_q = train[merge_on].drop_duplicates()
    train_q[f'validation_{k}'] = 0
    val_q = val[merge_on].drop_duplicates()
    val_q[f'validation_{k}'] = 1
    train_val_q = pd.concat([train_q, val_q], axis=0)
    df_list = df_list.merge(train_val_q, on=merge_on, how='left')
    df_list[f'validation_{k}'] = df_list[f'validation_{k}'].fillna(2)

    # Keep only train/validation rows if requested (drop validation==2)
    if val_only:
        df_list = df_list[df_list[f'validation_{k}'] != 2].reset_index(drop=True)
        print(f'val_only: kept {len(df_list)} rows (train + validation)')

    out_path = PREDICTIONS_DIR / f'mf_{split_type}_{k}_{n_splits}_{batch_size}_{dim}__resample{resample}_long.parquet'
    df_list.to_parquet(out_path)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    fire.Fire()
