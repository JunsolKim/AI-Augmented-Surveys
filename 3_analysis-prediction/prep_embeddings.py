import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pathlib import Path
import pickle

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
REPL = BASE
DATA = REPL / "data"
OUT = DATA / "fig_table_gen"


def _load_alpaca_embeddings() -> np.ndarray:
    cands = [
        DATA / "weights" / "maicomputer_alpaca-native.pkl",
        BASE / "data" / "weights" / "chavinlo_alpaca-native.pkl",
    ]
    for p in cands:
        if p.exists():
            with open(p, "rb") as f:
                e = pickle.load(f)
            return np.vstack(e) if isinstance(e, list) else np.asarray(e)
    raise FileNotFoundError(f"Alpaca embedding pkl not found in {cands}")


def _build_dcn(n_indiv: int, n_q: int, n_year: int, dim: int = 50):
    """Same architecture as step1_train.py."""
    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Reshape

    individual_id = Input(name="individual_id", shape=(1,))
    x1 = Embedding(n_indiv, dim, name="individual_embedding")(individual_id)
    x1 = Reshape((dim,), name="reshape1")(x1)

    question_id = Input(name="question_id", shape=(1,))
    q_emb = Embedding(*_load_alpaca_embeddings().shape, name="question_embedding")(question_id)
    q_emb.trainable = False
    x2 = Dense(dim, name="question_embedding2")(q_emb)
    x2 = Reshape((dim,), name="reshape2")(x2)

    year_id = Input(name="year_id", shape=(1,))
    x3 = Embedding(n_year, dim, name="year_embedding")(year_id)
    x3 = Reshape((dim,), name="reshape3")(x3)

    x = tf.concat([x1, x2, x3], axis=1, name="concat1")
    for i in range(3):
        x = tfrs.layers.dcn.Cross(
            projection_dim=dim * 3,
            kernel_initializer="glorot_uniform",
            name=f"cross_layer_{i}",
        )(x)
        x = Dropout(0.2)(x)
    for i in range(3):
        x = Dense(dim * 3, activation="relu", name=f"dense_layer_{i}")(x)
        x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid", name="out")(x)

    inp = {"individual_id": individual_id, "question_id": question_id, "year_id": year_id}
    return Model(inputs=inp, outputs=out)


def main() -> None:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(DATA / "df_analysis_slim.pkl").drop_duplicates(
        ["yearid_id", "question_id"]
    )
    raw = _load_alpaca_embeddings()

    weights_path = (
        DATA / "models"
        / "maicomputer_alpaca-native_partial_k0_10_128_50__resample1_best.weights.h5"
    )
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    model = _build_dcn(
        int(df["yearid_id"].max()) + 1,
        raw.shape[0],
        int(df["year_order"].max()) + 1,
    )
    model.get_layer("question_embedding").set_weights([raw])
    model.load_weights(str(weights_path))

    import tensorflow as tf

    ind_emb = model.get_layer("individual_embedding").get_weights()[0]
    yr_emb = model.get_layer("year_embedding").get_weights()[0]

    q_input = tf.keras.Input(shape=(1,))
    q = model.get_layer("question_embedding")(q_input)
    q = model.get_layer("question_embedding2")(q)
    q = model.get_layer("reshape2")(q)
    q_proj = tf.keras.Model(q_input, q).predict(np.arange(raw.shape[0]), verbose=0)

    # ---- questions ----
    used_qids = sorted(df["question_id"].unique())
    q_sub = q_proj[used_qids]
    q_xy = TSNE(perplexity=30, n_iter=2000, random_state=123).fit_transform(q_sub)
    q_km = KMeans(n_clusters=4, random_state=123, n_init=10).fit_predict(q_sub)
    var_lookup = dict(zip(df["question_id"], df["variable"]))
    pd.DataFrame({
        "question_id": used_qids,
        "variable": [var_lookup.get(qid, f"qid_{qid}") for qid in used_qids],
        "x": q_xy[:, 0], "y": q_xy[:, 1], "kmeans4": q_km,
    }).to_parquet(OUT / "embeddings_question.parquet", index=False)

    # ---- respondents ----
    used_rids = sorted(df["yearid_id"].unique())
    r_sub = ind_emb[used_rids]
    r_xy = TSNE(perplexity=30, n_iter=2000, random_state=123).fit_transform(r_sub)
    r_km = KMeans(n_clusters=10, random_state=123, n_init=10).fit_predict(r_sub)
    yr_lookup = dict(zip(df["yearid_id"], df["year"]))
    pd.DataFrame({
        "yearid_id": used_rids,
        "year": [yr_lookup.get(r) for r in used_rids],
        "x": r_xy[:, 0], "y": r_xy[:, 1], "kmeans10": r_km,
    }).to_parquet(OUT / "embeddings_respondent.parquet", index=False)

    # ---- periods ----
    used_orders = sorted(df["year_order"].unique())
    y_sub = yr_emb[used_orders]
    yr_to_real = dict(zip(df["year_order"], df["year"]))
    y_xy = TSNE(perplexity=2, n_iter=100000, random_state=123).fit_transform(y_sub)
    pd.DataFrame({
        "year_order": used_orders,
        "year": [yr_to_real.get(o) for o in used_orders],
        "x": y_xy[:, 0], "y": y_xy[:, 1],
    }).to_parquet(OUT / "embeddings_period.parquet", index=False)

    print(f"saved: {OUT/'embeddings_question.parquet'}")
    print(f"saved: {OUT/'embeddings_respondent.parquet'}")
    print(f"saved: {OUT/'embeddings_period.parquet'}")


if __name__ == "__main__":
    main()
