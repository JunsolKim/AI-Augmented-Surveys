import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

np.random.seed(1234)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
MODELS = DATA / "models"
OUT = DATA / "fig_table_gen"
OUT.mkdir(parents=True, exist_ok=True)

K = 3
N_SPLITS = 10
BATCH = 128
DIM = 50
MODEL_NAME = "maicomputer/alpaca-native"
TARGET_VAR = "marhomo1"
EXCLUDE_LEVELS = [0, 10, 20, 30, 40, 50]

sys.path.insert(0, str(BASE / "2_model-finetuning"))
from utils import load_data  # noqa: E402
from models import build_dcn_model  # noqa: E402


def weight_path(n_exclude):
    safe = MODEL_NAME.replace("/", "_")
    stem = f"{safe}_partial_k{K}_{N_SPLITS}_{BATCH}_{DIM}__resample1"
    if n_exclude > 0:
        stem = f"{stem}_exclude{n_exclude}"
    return MODELS / f"{stem}_best.weights.h5"


def main():
    import tensorflow as tf

    df_analysis, _train, val = load_data(
        split_type="partial", k=K, n_splits=N_SPLITS)
    val_t = val[val["variable"] == TARGET_VAR].copy()
    if len(val_t) == 0:
        raise RuntimeError(
            f"{TARGET_VAR} not present in k={K} validation set "
            "— check fold seed.")
    if val_t["binarized"].nunique() < 2:
        raise RuntimeError(
            f"{TARGET_VAR} validation rows are single-class "
            "— cannot compute AUC.")
    print(f"k={K} validation rows for {TARGET_VAR}: {len(val_t):,} "
          f"(pos={(val_t['binarized']==1).sum():,}, "
          f"neg={(val_t['binarized']==0).sum():,})")

    feed = {
        "individual_id": np.array(val_t["yearid_id"]),
        "question_id":   np.array(val_t["question_id"]),
        "year_id":       np.array(val_t["year_order"]),
    }
    y_true = np.array(val_t["binarized"]).astype(int)

    model = build_dcn_model(MODEL_NAME, df_analysis, dim=DIM)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy()],
    )

    rows = []
    for n in EXCLUDE_LEVELS:
        wp = weight_path(n)
        if not wp.exists():
            print(f"[MISS] n_exclude={n}: {wp.name}")
            continue
        model.load_weights(str(wp))
        p = model.predict(feed, batch_size=4096, verbose=0).squeeze()
        auc = float(roc_auc_score(y_true, p))
        acc = float(accuracy_score(y_true, (p >= 0.5).astype(int)))
        rows.append({"n_exclude": int(n), "auc": auc, "accuracy": acc,
                     "n": int(len(y_true))})
        print(f"  n_exclude={n:2d}: AUC={auc:.4f}  Acc={acc:.4f}  "
              f"n={len(y_true):,}")

    if not rows:
        raise RuntimeError(
            "No weights loaded — cannot emit samesex_exclude.parquet")

    out = OUT / "samesex_exclude.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
