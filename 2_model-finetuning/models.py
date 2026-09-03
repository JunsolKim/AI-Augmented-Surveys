import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Reshape


BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "data" / "weights"


def build_dcn_model(model_name, df_analysis, dim=50, layer=None):
    """Build the DCN prediction model.

    Args:
        model_name: LLM name (e.g. 'chavinlo/alpaca-native') — loads embeddings from weights dir
        df_analysis: DataFrame with yearid_id, question_id, year_order columns
        dim: embedding dimension for individual/year embeddings
        layer: if specified, loads middle-layer embeddings ({model}_layer{layer}.pkl)

    Returns:
        Compiled keras Model with inputs: {individual_id, question_id, year_id}
    """
    safe_name = model_name.replace('/', '_')
    if layer is not None:
        weights_path = WEIGHTS_DIR / f'{safe_name}_layer{layer}.pkl'
    else:
        weights_path = WEIGHTS_DIR / f'{safe_name}.pkl'
    weights = pickle.load(open(weights_path, 'rb'))
    if isinstance(weights, list):
        weights = np.vstack(weights)

    n_individuals = int(df_analysis['yearid_id'].max()) + 1
    n_questions = weights.shape[0]
    q_embed_dim = weights.shape[1]
    n_years = int(df_analysis['year_order'].max()) + 1

    individual_id = Input(name='individual_id', shape=(1,))
    x1 = Embedding(n_individuals, dim, name='individual_embedding')(individual_id)
    x1 = Reshape((dim,), name='reshape1')(x1)

    question_id = Input(name='question_id', shape=(1,))
    question_embedding = Embedding(n_questions, q_embed_dim, weights=[weights],
                                    name='question_embedding')(question_id)
    question_embedding.trainable = False
    x2 = Dense(50, name='question_embedding2')(question_embedding)
    x2 = Reshape((50,), name='reshape2')(x2)

    year_id = Input(name='year_id', shape=(1,))
    x3 = Embedding(n_years, dim, name='year_embedding')(year_id)
    x3 = Reshape((dim,), name='reshape3')(x3)

    x = tf.concat([x1, x2, x3], axis=1, name='concat1')
    for i in range(3):
        x = tfrs.layers.dcn.Cross(projection_dim=dim * 3, kernel_initializer="glorot_uniform",
                                   name=f'cross_layer_{i}')(x)
        x = Dropout(0.2)(x)
    for i in range(3):
        x = Dense(dim * 3, activation="relu", name=f'dense_layer_{i}')(x)
        x = Dropout(0.2)(x)

    out = Dense(1, activation='sigmoid', name="out", dtype='float32')(x)
    inputs = {'individual_id': individual_id, 'question_id': question_id, 'year_id': year_id}
    model = Model(inputs=inputs, outputs=out)
    return model


def load_trained_model(model_name, df_analysis, weight_path, dim=50, layer=None):
    """Build model and load trained weights."""
    model = build_dcn_model(model_name, df_analysis, dim=dim, layer=layer)
    model.load_weights(weight_path)
    return model


class ExplicitALS:
    """Weighted Alternating Least Squares matrix factorization baseline."""

    def __init__(self):
        os.environ['PYTHONHASHSEED'] = str(42)
        random.seed(42)
        np.random.seed(42)

    def fit(self, ratings, C, val, n_factors=50, r_lambda=10, n_iter=15):
        from scipy.sparse import diags

        self.ratings = ratings
        self.C = C
        self.n_users = ratings.shape[0]
        self.n_items = ratings.shape[1]
        self.n_factors = n_factors
        self.r_lambda = r_lambda

        X = np.random.normal(scale=1. / self.n_factors, size=(self.n_users, self.n_factors))
        Y = np.random.normal(scale=1. / self.n_factors, size=(self.n_items, self.n_factors))
        lambda_I = self.r_lambda * np.identity(self.n_factors)

        for epoch in range(n_iter):
            print(f"Epoch {epoch}:")
            # Optimize users
            for u in tqdm(range(self.n_users), desc='Users'):
                cU = diags(self.C[u].todense().A1)
                pU = self.ratings[u, :].todense().A1
                yT_cU = cU.T.dot(Y).T
                X[u] = np.linalg.solve(yT_cU.dot(Y) + lambda_I, yT_cU.dot(pU))
            # Optimize items
            for i in tqdm(range(self.n_items), desc='Items'):
                cI = diags(self.C[:, i].todense().A1)
                pI = self.ratings[:, i].todense().A1
                xT_cI = cI.T.dot(X).T
                Y[i] = np.linalg.solve(xT_cI.dot(X) + lambda_I, xT_cI.dot(pI))

            pred = X.dot(Y.T)
            val_pred = pred[val.yearid_id, val.question_id]
            print(f'  AUC={roc_auc_score(val.binarized, val_pred):.4f}')

        self.user_factors = X
        self.item_factors = Y
