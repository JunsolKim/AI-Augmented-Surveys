from __future__ import annotations

import os
import random
import time
from multiprocessing import get_context, shared_memory
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.linalg import solve_triangular
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from utils import load_data

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PREDICTIONS_DIR = DATA_DIR / "predictions"
CORR_CACHE = DATA_DIR / "cache" / "pairwise_corr_full.parquet"


# ---------------------------------------------------------------------------
# Shared-memory globals populated in each worker by _worker_init.
# ---------------------------------------------------------------------------
_STATE: dict = {}


def _worker_init(spec: dict) -> None:
    """Attach to the parent's shared-memory blocks (called once per worker)."""
    global _STATE
    refs = {}
    bufs = {}
    for key, (name, shape, dtype) in spec.items():
        shm = shared_memory.SharedMemory(name=name)
        refs[key] = shm
        bufs[key] = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
    _STATE["refs"] = refs
    _STATE["buf"] = bufs


def _impute_column(args: tuple) -> tuple:
    """van Buuren `mice.impute.logreg`: posterior-draw logistic + Bernoulli sample."""
    q, m_draws, seed = args
    mat = _STATE["buf"]["mat"]
    mat_work = _STATE["buf"]["mat_work"]
    top_feat = _STATE["buf"]["top_feat"]
    col_mean = _STATE["buf"]["col_mean"]
    n_users = mat.shape[0]

    rng = np.random.default_rng(seed + q)
    y = mat[:, q]
    ry = np.isfinite(y)
    n_obs = int(ry.sum())
    out = np.empty(n_users, dtype=np.float64)

    if n_obs < 10:
        out.fill(col_mean[q])
        return q, out

    y_bin_obs = np.rint(y[ry]).astype(np.int64)
    uniq = np.unique(y_bin_obs)
    if uniq.size < 2:
        out.fill(float(uniq[0]))
        return q, out

    feat_idx = top_feat[q]
    X_all = mat_work[:, feat_idx]
    X_obs = X_all[ry]
    miss_idx = np.where(~ry)[0]
    n_miss = miss_idx.size

    # Observed cells: keep actual 0/1 label.
    out[ry] = y_bin_obs.astype(np.float64)
    if n_miss == 0:
        return q, out

    try:
        lr = LogisticRegression(C=1e8, max_iter=200, solver="lbfgs", n_jobs=1)
        lr.fit(X_obs, y_bin_obs)
    except Exception:
        out[miss_idx] = col_mean[q]
        return q, out
    beta_hat = np.concatenate([lr.intercept_, lr.coef_[0]])

    # Fisher information H = X^T W X with W = p(1-p).
    p_obs = lr.predict_proba(X_obs)[:, 1]
    W = p_obs * (1.0 - p_obs)
    X_obs_aug = np.empty((n_obs, X_obs.shape[1] + 1), dtype=np.float64)
    X_obs_aug[:, 0] = 1.0
    X_obs_aug[:, 1:] = X_obs
    H = (X_obs_aug.T * W) @ X_obs_aug
    try:
        L = np.linalg.cholesky(H + 1e-6 * np.eye(H.shape[0]))
    except np.linalg.LinAlgError:
        out[miss_idx] = col_mean[q]
        return q, out

    X_miss_aug = np.empty((n_miss, X_obs.shape[1] + 1), dtype=np.float64)
    X_miss_aug[:, 0] = 1.0
    X_miss_aug[:, 1:] = X_all[miss_idx]
    fill_sum = np.zeros(n_miss, dtype=np.float64)
    for _ in range(m_draws):
        z = rng.standard_normal(beta_hat.size)
        u = solve_triangular(L, z, lower=True, check_finite=False)
        delta = solve_triangular(L.T, u, lower=False, check_finite=False)
        beta_star = beta_hat + delta
        p_miss = expit(X_miss_aug @ beta_star)
        y_draw = rng.binomial(1, p_miss)
        fill_sum += y_draw
    out[miss_idx] = fill_sum / m_draws
    return q, out


# ---------------------------------------------------------------------------
# Helpers for shared-memory setup.
# ---------------------------------------------------------------------------
def _make_shm(arr: np.ndarray):
    """Create a SharedMemory block and copy arr's contents into it."""
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[:] = arr
    return shm, view


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(split_type="impute", k=0, n_splits=10, resample=1,
        use_demo=False, use_poli=True,
        maxit=15, n_feat=40, m_draws=5,
        n_cores=20, seed=42,
        batch_size=128, dim=50,
        chunksize=10):
    """Run proper-MICE imputation for one fold."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = (
        f"mice_pyr_{split_type}_{k}_{n_splits}_{batch_size}_{dim}__resample{resample}_long.parquet"
    )
    out_path = PREDICTIONS_DIR / out_name
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing {out_path}; delete it first if you really mean to.")

    t0 = time.time()
    df_analysis, train, val = load_data(
        split_type=split_type, k=k, n_splits=n_splits, resample=resample,
        use_demo=use_demo, use_poli=use_poli,
    )
    n_users = int(df_analysis["yearid_id"].max()) + 1
    n_items = int(df_analysis["question_id"].max()) + 1
    print(f"[{time.strftime('%H:%M:%S')}] Matrix: {n_users:,} users x {n_items:,} items  "
          f"n_cores={n_cores} maxit={maxit} n_feat={n_feat} m_draws={m_draws}")

    # -------- Train-only dense matrix --------
    mat = np.full((n_users, n_items), np.nan, dtype=np.float64)
    mat[train["yearid_id"].values, train["question_id"].values] = train["binarized"].values
    obs = np.isfinite(mat)
    print(f"Observed cells: {obs.sum():,} / {mat.size:,} ({obs.mean():.2%})")

    # -------- Top-N predictors from precomputed pairwise-complete corr --------
    if not CORR_CACHE.exists():
        raise SystemExit(f"Missing precomputed corr matrix: {CORR_CACHE}")
    print(f"Loading {CORR_CACHE.name} ...")
    corr = pd.read_parquet(CORR_CACHE).to_numpy(dtype=np.float64)
    corr = np.nan_to_num(corr, nan=0.0)
    assert corr.shape == (n_items, n_items), f"Corr shape {corr.shape} != ({n_items}, {n_items})"
    top_feat = np.empty((n_items, n_feat), dtype=np.int64)
    for q in range(n_items):
        s = np.abs(corr[q])
        s[q] = -1.0
        top_feat[q] = np.argpartition(-s, n_feat)[:n_feat]
    del corr
    print(f"Top-{n_feat} predictors selected per column.")

    # -------- Initial fill with column means --------
    col_mean_full = np.nanmean(mat, axis=0)
    col_mean_full = np.nan_to_num(col_mean_full, nan=0.5)
    mat_work = np.where(obs, mat, col_mean_full[np.newaxis, :])

    # -------- Create shared memory blocks once --------
    mat_shm,      mat_sh      = _make_shm(mat)
    matwork_shm,  mat_work_sh = _make_shm(mat_work)
    topfeat_shm,  top_feat_sh = _make_shm(top_feat)
    colmean_shm,  col_mean_sh = _make_shm(col_mean_full)
    # Release the python-side duplicates; workers attach by name.
    del mat, mat_work, top_feat, col_mean_full

    spec = {
        "mat":      (mat_shm.name,     mat_sh.shape,      str(mat_sh.dtype)),
        "mat_work": (matwork_shm.name, mat_work_sh.shape, str(mat_work_sh.dtype)),
        "top_feat": (topfeat_shm.name, top_feat_sh.shape, str(top_feat_sh.dtype)),
        "col_mean": (colmean_shm.name, col_mean_sh.shape, str(col_mean_sh.dtype)),
    }

    val_yid = val["yearid_id"].values
    val_qid = val["question_id"].values
    val_true = val["binarized"].values
    val_ok_mask = np.isfinite(val_true)

    pred = np.empty_like(mat_sh)  # separate buffer for this iteration's output

    # Use fork so worker importation cost is zero; shared-memory writes propagate automatically.
    ctx = get_context("fork")
    print(f"[{time.strftime('%H:%M:%S')}] Starting Pool(n_cores={n_cores}) ...")

    try:
        with ctx.Pool(n_cores, initializer=_worker_init, initargs=(spec,)) as pool:
            for it in range(1, maxit + 1):
                t_iter = time.time()
                tasks = [(q, m_draws, seed + 1000 * it) for q in range(n_items)]
                pbar = tqdm(total=n_items, desc=f"iter {it:2d}/{maxit}",
                            mininterval=1.0, ncols=100)
                for q, col in pool.imap_unordered(_impute_column, tasks, chunksize=chunksize):
                    pred[:, q] = col
                    miss_mask = ~obs[:, q]
                    # Write back into the SHARED mat_work so workers pick up new values
                    # on the next iteration without any pickling.
                    mat_work_sh[miss_mask, q] = col[miss_mask]
                    pbar.update(1)
                pbar.close()
                t_elapsed = time.time() - t_iter

                # Validation diagnostic
                p_val = pred[val_yid, val_qid]
                ok = val_ok_mask & np.isfinite(p_val)
                try:
                    auc = roc_auc_score(val_true[ok], p_val[ok]) if ok.sum() > 1 else float("nan")
                    acc = float(((p_val[ok] >= 0.5) == (val_true[ok] == 1)).mean())
                except ValueError:
                    auc = float("nan"); acc = float("nan")
                print(f"  iter {it:2d}: {t_elapsed:5.1f}s  AUC={auc:.4f}  Acc={acc:.4f}  n_val={int(ok.sum()):,}")

        print(f"\nAll iterations done in {time.time() - t0:.0f}s. Streaming long parquet ...")

        # -------- Stream long parquet (one row group per question) --------
        yo = df_analysis[["yearid_id", "year_order"]].drop_duplicates().reset_index(drop=True)
        if split_type in ("impute", "mar", "mnar"):
            merge_on = ["question_id", "yearid_id"]
        else:
            merge_on = ["question_id", "year_order"]
        train_q = train[merge_on].drop_duplicates().copy()
        train_q[f"validation_{k}"] = 0
        val_q = val[merge_on].drop_duplicates().copy()
        val_q[f"validation_{k}"] = 1
        tv = pd.concat([train_q, val_q], axis=0, ignore_index=True)

        schema = pa.schema([
            ("yearid_id", pa.int64()),
            ("year_order", pa.int64()),
            ("question_id", pa.int64()),
            (f"response_{k}", pa.float64()),
            (f"validation_{k}", pa.int64()),
        ])
        # Stream to a temp name and rename on success.
        tmp_path = out_path.with_suffix(".parquet.tmp")
        writer = pq.ParquetWriter(tmp_path, schema, compression="snappy")
        try:
            for q in tqdm(range(n_items), desc="write", ncols=100):
                chunk = yo.copy()
                chunk["question_id"] = np.int64(q)
                chunk[f"response_{k}"] = pred[chunk["yearid_id"].values, q]
                chunk = chunk.merge(tv, on=merge_on, how="left")
                chunk[f"validation_{k}"] = chunk[f"validation_{k}"].fillna(2).astype(np.int64)
                chunk = chunk[["yearid_id", "year_order", "question_id", f"response_{k}", f"validation_{k}"]]
                writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
        finally:
            writer.close()
        tmp_path.rename(out_path)
        print(f"Saved -> {out_path}  ({time.time() - t0:.0f}s total)")

    finally:
        # Clean up shared memory blocks.
        for shm in (mat_shm, matwork_shm, topfeat_shm, colmean_shm):
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    fire.Fire()
