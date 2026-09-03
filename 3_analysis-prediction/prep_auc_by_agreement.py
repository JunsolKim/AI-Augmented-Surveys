from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

np.random.seed(1234)
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_gen"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading observations ...")
    df = pd.read_pickle(DATA / "df_analysis_slim.pkl").drop_duplicates(
        ["yearid_id", "question_id"])

    print("Loading impute wide (bare pred cols only) ...")
    wide_path = PRED / "maicomputer_alpaca-native_impute_10_128_50__resample1_wide.parquet"
    all_cols = [f.name for f in pq.read_schema(wide_path)]
    suf = ("_obs_bin", "_rescale", "_rescale_logit",
           "_rescale_glm", "_rescale_logit_glm")
    bare = [c for c in all_cols
            if c != "yearid_id" and not any(c.endswith(s) for s in suf)]
    print(f"  bare pred cols: {len(bare)}")
    wide = pd.read_parquet(wide_path, columns=["yearid_id"] + bare)
    obs_by_var = df[["yearid_id", "variable", "binarized"]].groupby("variable")

    rows = []
    for i, var in enumerate(bare):
        if var not in obs_by_var.groups:
            continue
        obs_sub = obs_by_var.get_group(var)[["yearid_id", "binarized"]]
        pred_sub = wide[["yearid_id", var]].rename(columns={var: "predicted"})
        m = obs_sub.merge(pred_sub, on="yearid_id", how="inner").dropna(
            subset=["binarized", "predicted"])
        n_pos = int(m["binarized"].sum())
        n_neg = int(len(m) - n_pos)
        if n_pos < 10 or n_neg < 10 or len(m) < 30:
            continue
        auc = float(roc_auc_score(m["binarized"], m["predicted"]))
        rows.append({
            "variable": var,
            "pct_positive": float(m["binarized"].mean()),
            "auc": auc,
            "n": int(len(m)),
        })
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(bare)} vars")

    out = OUT / "auc_by_agreement.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"Saved: {out} ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
