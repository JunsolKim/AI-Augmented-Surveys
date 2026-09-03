import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import pickle
import random
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from utils import load_data
from models import build_dcn_model, ExplicitALS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "data" / "models"


def _weight_name(model_name, split_type, k, n_splits, batch_size, dim, resample, use_demo, use_poli, layer=None, n_exclude=0, max_q=0):
    safe = model_name.replace('/', '_')
    base = f'{safe}_{split_type}_k{k}_{n_splits}_{batch_size}_{dim}__resample{resample}'
    if layer is not None:
        base = f'{base}_layer{layer}'
    if n_exclude > 0:
        base = f'{base}_exclude{n_exclude}'
    if max_q > 0:
        base = f'{base}_nq{max_q}'
    if use_demo is False and use_poli is True:
        return base
    return f'{base}_{use_demo}_{use_poli}'


def dcn(model_name='chavinlo/alpaca-native', split_type='partial', k=0, n_splits=10,
        resample=1, batch_size=128, dim=50, use_demo=False, use_poli=True, epochs=65, layer=None,
        n_exclude=0, max_q=0):
    """Train DCN model for one fold."""
    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load exclude list if needed (top N correlated vars with marhomo1)
    exclude_vars = None
    if n_exclude > 0:
        corr_df = pd.read_parquet(DATA_DIR / 'corr_with_marhomo1.parquet')
        exclude_vars = corr_df.head(n_exclude)['variable'].tolist()
        print(f'Excluding {len(exclude_vars)} variables from training')

    df_analysis, train, val = load_data(
        split_type=split_type, k=k, n_splits=n_splits,
        resample=resample, use_demo=use_demo, use_poli=use_poli,
        exclude_vars=exclude_vars, max_q=max_q,
    )

    train_features = {
        'individual_id': np.array(train['yearid_id']),
        'question_id': np.array(train['question_id']),
        'year_id': np.array(train['year_order']),
    }
    train_label = np.array(train['binarized'])
    val_features = {
        'individual_id': np.array(val['yearid_id']),
        'question_id': np.array(val['question_id']),
        'year_id': np.array(val['year_order']),
    }
    val_label = np.array(val['binarized'])

    wname = _weight_name(model_name, split_type, k, n_splits, batch_size, dim, resample, use_demo, use_poli, layer=layer, n_exclude=n_exclude, max_q=max_q)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = build_dcn_model(model_name, df_analysis, dim=dim, layer=layer)

        checkpoint = ModelCheckpoint(
            filepath=str(MODELS_DIR / f'{wname}_best.weights.h5'),
            save_freq='epoch', verbose=1, monitor='val_loss',
            save_best_only=True, save_weights_only=True,
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            2e-05, decay_steps=80000, decay_rate=0.96, staircase=True
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy()],
        )

        if split_type == 'total':
            epochs = min(epochs, 10)

        history = model.fit(
            train_features, train_label,
            validation_data=(val_features, val_label),
            batch_size=batch_size, epochs=epochs, verbose=1,
            callbacks=[checkpoint],
            use_multiprocessing=True, workers=20,
        )

    # Save final weights and history
    model.save_weights(str(MODELS_DIR / f'{wname}_final.weights.h5'))
    pickle.dump(history.history, open(MODELS_DIR / f'history_{wname}.pkl', 'wb'))
    print(f'Done: {wname}')


def mf(split_type='partial', k=0, n_splits=10, resample=1, dim=50, use_demo=False, use_poli=True):
    """Train matrix factorization baseline for one fold."""
    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    np.random.seed(42)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_analysis, train, val = load_data(
        split_type=split_type, k=k, n_splits=n_splits,
        resample=resample, use_demo=use_demo, use_poli=use_poli,
    )

    n_users = int(df_analysis['yearid_id'].max()) + 1
    n_items = int(df_analysis['question_id'].max()) + 1

    ratings = csr_matrix(
        (np.array(train['binarized']), (np.array(train['yearid_id']), np.array(train['question_id']))),
        shape=(n_users, n_items),
    )
    C = csr_matrix(
        (np.ones(len(train)), (np.array(train['yearid_id']), np.array(train['question_id']))),
        shape=(n_users, n_items),
    )

    model_mf = ExplicitALS()
    model_mf.fit(ratings, C, val, n_factors=dim, n_iter=15)

    out_path = MODELS_DIR / f'mf_{split_type}_k{k}_{n_splits}_{dim}__resample{resample}.pkl'
    pickle.dump(model_mf, open(out_path, 'wb'))
    print(f'Saved MF model -> {out_path}')


if __name__ == '__main__':
    fire.Fire()
