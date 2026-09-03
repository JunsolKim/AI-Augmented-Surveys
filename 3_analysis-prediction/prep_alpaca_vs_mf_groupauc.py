import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_gen"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "3_analysis-prediction"))
from prep_individualauc import build_demographics  # noqa: E402


DEMO_COLS = ["sex", "race", "degree", "region4", "income_q5",
             "age_group", "period_group", "partyid7"]
TASKS = [
    ("impute", "Missing Data Imputation"),
    ("partial", "Retrodiction"),
]
MODELS = [
    ("Alpaca-7b", "maicomputer_alpaca-native"),
    ("MF",        "mf"),
]
SUFFIXES = ("_obs_bin", "_rescale", "_rescale_logit",
            "_rescale_glm", "_rescale_logit_glm")
NON_PRED_COLS = {"yearid_id", "id", "wtssall", "wtssnr", "wtssnrps"}


def _bare_pred_cols(parquet_path):
    cols = [f.name for f in pq.read_schema(parquet_path)]
    return [c for c in cols if c not in NON_PRED_COLS
            and not any(c.endswith(s) for s in SUFFIXES)]


def long_obs_pred(wide_path, df_obs):
    """Return a long (yearid_id, binarized, pred) frame: the intersection
    of df_obs and the wide parquet's variables, with predictions looked
    up via vectorised numpy indexing (same recipe as
    prep_fig_alpaca_vs_mf.py::individual_aucs)."""
    bare = _bare_pred_cols(wide_path)
    print(f"    loading {wide_path.name}  ({len(bare)} pred cols)")
    wide = pd.read_parquet(wide_path, columns=["yearid_id"] + bare)
    wide = wide.set_index("yearid_id")
    wide_arr = wide.to_numpy(copy=False)
    var_to_col = {v: j for j, v in enumerate(bare)}
    yid_to_row = pd.Series(np.arange(len(wide)), index=wide.index)

    sub = df_obs[df_obs["variable"].isin(var_to_col)]
    row_idx = yid_to_row.reindex(sub["yearid_id"].values).to_numpy()
    col_idx = sub["variable"].map(var_to_col).to_numpy()
    keep = pd.notna(row_idx) & pd.notna(col_idx)
    row_idx = row_idx[keep].astype(np.int64)
    col_idx = col_idx[keep].astype(np.int64)
    yids = sub["yearid_id"].to_numpy()[keep]
    bins = sub["binarized"].to_numpy()[keep]

    pred = wide_arr[row_idx, col_idx]
    finite = np.isfinite(pred) & ~pd.isna(bins)
    return pd.DataFrame({
        "yearid_id": yids[finite],
        "binarized": bins[finite].astype(np.int8),
        "pred":      pred[finite],
    })


def main() -> None:
    t0 = time.time()
    print("Loading observations ...")
    df = (pd.read_pickle(DATA / "df_analysis_slim.pkl")
          .drop_duplicates(["yearid_id", "question_id"]))
    df_obs = (df[["yearid_id", "yearid", "variable", "binarized"]]
              .dropna(subset=["binarized"]))
    yid_to_yearid = df_obs.drop_duplicates("yearid_id")[["yearid_id", "yearid"]]
    print(f"  obs rows: {len(df_obs):,}")

    print("Building demographics (original R bins) ...")
    demo = build_demographics()
    yid_demo = yid_to_yearid.merge(demo, on="yearid", how="left")
    yid_demo = yid_demo[["yearid_id"] + DEMO_COLS].dropna(subset=DEMO_COLS)
    print(f"  yid x demo rows (complete): {len(yid_demo):,}")

    rows = []
    for split, task_label in TASKS:
        for model_label, stem in MODELS:
            wide_path = PRED / f"{stem}_{split}_10_128_50__resample1_wide.parquet"
            if not wide_path.exists():
                print(f"  [MISS] {wide_path.name}")
                continue
            print(f"\n=== {model_label} | {task_label} ===")
            lp = long_obs_pred(wide_path, df_obs)
            lp = lp.merge(yid_demo, on="yearid_id", how="inner")
            print(f"    obs after demo join: {len(lp):,}")

            for g in DEMO_COLS:
                for lvl, sub in lp.groupby(g, observed=True):
                    n_obs = len(sub)
                    n_pos = int(sub["binarized"].sum())
                    n_neg = n_obs - n_pos
                    if n_obs < 100 or n_pos < 10 or n_neg < 10:
                        continue
                    auc = roc_auc_score(sub["binarized"].to_numpy(),
                                        sub["pred"].to_numpy())
                    rows.append({
                        "task":          task_label,
                        "model":         model_label,
                        "demo_group":    g,
                        "subgroup":      lvl,
                        "n_respondents": int(sub["yearid_id"].nunique()),
                        "n_obs":         n_obs,
                        "auc":           float(auc),
                    })

    out = pd.DataFrame(rows)
    out_path = OUT / "alpaca_vs_mf_groupauc.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(out)} rows)  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
