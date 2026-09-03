from pathlib import Path
import pickle

import fire
import h5py
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = DATA / "fig_table_gen"

TARGET_VARS = ["homosex", "marhomo1", "busing", "nomeat"]


def _load_dense_weights(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract the (4096 -> 50) Dense projection from a trained DCN checkpoint.

    Layer order in the saved Keras model: embedding (alpaca), embedding_1
    (individual), embedding_2 (year), then dense (question_embedding2:
    4096->50). We identify this layer by its unique shape rather than name.
    """
    with h5py.File(h5_path, "r") as f:
        layers = f["layers"]
        for name in layers.keys():
            vars_ = layers[name].get("vars")
            if vars_ is None or "0" not in vars_:
                continue
            w = vars_["0"][...]
            if w.shape == (4096, 50):
                b = vars_["1"][...] if "1" in vars_ else np.zeros(50, dtype=w.dtype)
                return w.astype(np.float32), b.astype(np.float32)
    raise RuntimeError(f"Dense(4096,50) layer not found in {h5_path}")


def _load_raw_embeddings(pkl_path: Path) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        emb = pickle.load(f)
    if isinstance(emb, list):
        emb = np.vstack(emb)
    return np.asarray(emb, dtype=np.float32)


def main(top_k: int = 3, fold: int = 0):
    OUT.mkdir(parents=True, exist_ok=True)

    raw_pkl = DATA / "weights" / "maicomputer_alpaca-native.pkl"
    dcn_h5 = (DATA / "models" /
              f"maicomputer_alpaca-native_partial_k{fold}_10_128_50__resample1_best.weights.h5")
    if not raw_pkl.exists():
        raise FileNotFoundError(raw_pkl)
    if not dcn_h5.exists():
        raise FileNotFoundError(dcn_h5)

    raw = _load_raw_embeddings(raw_pkl)                       # (3603, 4096)
    W, b = _load_dense_weights(dcn_h5)                        # (4096, 50), (50,)
    emb = raw @ W + b                                         # (3603, 50)
    print(f"raw alpaca: {raw.shape}  ->  fine-tuned 50-dim: {emb.shape}")

    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    emb_n = emb / norms

    df = (pd.read_pickle(DATA / "df_analysis_slim.pkl")
            .drop_duplicates(["variable", "question_id"]))[["variable", "question_id"]]
    var_qid = dict(zip(df["variable"], df["question_id"].astype(int)))
    qid_var = {int(v): k for k, v in var_qid.items()}

    vm = pd.read_parquet(DATA / "gss_train_vars_corrected_binarized_again.parquet")
    qtext = dict(zip(vm["variable_name"], vm["question"].fillna("")))

    rows = []
    for tv in TARGET_VARS:
        qi = var_qid.get(tv)
        if qi is None or qi >= len(emb_n):
            print(f"[skip] {tv}: qid missing or out of range")
            continue
        sims = emb_n @ emb_n[qi]
        order = np.argsort(-sims)
        top = [j for j in order if j != qi][:top_k]
        for rank, j in enumerate(top, 1):
            v2 = qid_var.get(int(j), f"qid_{int(j)}")
            rows.append({
                "target": tv,
                "target_question": qtext.get(tv, ""),
                "rank": rank,
                "similar_variable": v2,
                "similar_question": qtext.get(v2, ""),
                "cosine": float(sims[j]),
            })
            print(f"  {tv} -> {v2} (rank={rank}, cos={sims[j]:.3f})")

    out = OUT / "similar_questions.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\nSaved: {out} ({len(rows)} rows)")


if __name__ == "__main__":
    fire.Fire(main)
