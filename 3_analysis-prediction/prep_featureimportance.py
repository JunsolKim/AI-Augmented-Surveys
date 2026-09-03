import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import KFold


np.random.seed(1234)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = DATA / "fig_table_gen"

K = 2
N_SPLITS = 10
DIM = 50

CONDITIONS = [
    ("Original",  None),
    ("Shuffle Q", "question_id"),
    ("Shuffle R", "individual_id"),
    ("Shuffle P", "year_id"),
]


def _build_model(n_individuals: int, n_years: int, q_emb: np.ndarray):
    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Reshape

    ii = Input(name="individual_id", shape=(1,))
    x1 = Embedding(n_individuals, DIM, name="individual_embedding")(ii)
    x1 = Reshape((DIM,), name="reshape1")(x1)

    qi = Input(name="question_id", shape=(1,))
    q = Embedding(q_emb.shape[0], q_emb.shape[1], name="question_embedding")(qi)
    q.trainable = False
    x2 = Dense(50, name="question_embedding2")(q)
    x2 = Reshape((50,), name="reshape2")(x2)

    yi = Input(name="year_id", shape=(1,))
    x3 = Embedding(n_years, DIM, name="year_embedding")(yi)
    x3 = Reshape((DIM,), name="reshape3")(x3)

    x = tf.concat([x1, x2, x3], axis=1, name="concat1")
    for i in range(3):
        x = tfrs.layers.dcn.Cross(projection_dim=DIM * 3,
             kernel_initializer="glorot_uniform", name=f"cross_layer_{i}")(x)
        x = Dropout(0.2)(x)
    for i in range(3):
        x = Dense(DIM * 3, activation="relu", name=f"dense_layer_{i}")(x)
        x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid", name="out")(x)
    return Model(inputs={"individual_id": ii, "question_id": qi, "year_id": yi},
                 outputs=out)


def _partial_val_split(df: pd.DataFrame, k: int, n_splits: int) -> pd.DataFrame:
    """Hold out k-th fold of (question_id, year_order) pairs with count > 1."""
    unique_q = df[["question_id", "year_order"]].drop_duplicates().reset_index(drop=True)
    unique_q["count"] = unique_q.groupby("question_id").transform("count")
    two = unique_q.loc[unique_q["count"] > 1].reset_index(drop=True)
    kfold = KFold(shuffle=True, n_splits=n_splits, random_state=42)
    val_idx = list(kfold.split(two))[k][1]
    val_q = two.iloc[val_idx].drop(columns=["count"]).copy()
    val_q["validation"] = 1
    df = df.merge(val_q, on=["question_id", "year_order"], how="left")
    df["validation"] = df["validation"].fillna(0).astype(int)
    return df[df["validation"] == 1].copy()


def _metrics(label: str, y: np.ndarray, p: np.ndarray) -> dict:
    yhat = (p >= 0.5).astype(int)
    auc = float(roc_auc_score(y, p))
    acc = float(accuracy_score(y, yhat))
    f1 = float(f1_score(y, yhat))
    print(f"  [{label:<10}] AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}")
    return {"condition": label, "auc": auc, "accuracy": acc, "f1": f1, "n": int(len(y))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(DATA / "df_analysis.pkl")
    df = df.loc[df["binarized"].notna() & df["yearid_id"].notna() & df["question_id"].notna()]
    df = df.drop_duplicates(["yearid_id", "question_id"])

    q_emb = pickle.load(open(DATA / "weights" / "maicomputer_alpaca-native.pkl", "rb"))
    q_emb = np.vstack(q_emb) if isinstance(q_emb, list) else np.asarray(q_emb)

    val = _partial_val_split(df, K, N_SPLITS)
    y_true = val["binarized"].astype(int).to_numpy()
    print(f"val rows: {len(val):,}")

    model = _build_model(
        n_individuals=int(df["yearid_id"].max()) + 1,
        n_years=int(df["year_order"].max()) + 1,
        q_emb=q_emb,
    )
    model.get_layer("question_embedding").set_weights([q_emb])
    wpath = DATA / "models" / (
        f"maicomputer_alpaca-native_partial_k{K}_{N_SPLITS}_128_{DIM}"
        f"__resample1_best.weights.h5"
    )
    assert wpath.exists(), f"missing: {wpath}"
    model.load_weights(str(wpath))
    print("model loaded.")

    feed = {
        "individual_id": val["yearid_id"].to_numpy(),
        "question_id":   val["question_id"].to_numpy(),
        "year_id":       val["year_order"].to_numpy(),
    }
    rows = []
    for label, shuf_key in CONDITIONS:
        if shuf_key is None:
            f = feed
        else:
            perm = np.random.permutation(len(feed[shuf_key]))
            f = {k_: (v[perm] if k_ == shuf_key else v) for k_, v in feed.items()}
        p = model.predict(f, batch_size=4096, verbose=0).squeeze()
        rows.append(_metrics(label, y_true, p))

    out_path = OUT / "featureimportance_permutation.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
