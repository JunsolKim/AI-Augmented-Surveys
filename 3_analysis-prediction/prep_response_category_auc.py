from pathlib import Path
import re
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


def count_categories(code_labels):
    """Count real response options in GSS `code_labels`.

    Lines starting with `.` are missing codes (`.i`, `.n`, `.d`, `.r`, `.s`,
    ...) and MUST be excluded — not substantive categories.
    """
    if pd.isna(code_labels) or not isinstance(code_labels, str):
        return None
    parts = re.split(r"[\n,;|]", code_labels)
    parts = [p.strip() for p in parts if p.strip()
             and not p.strip().startswith(".")]
    return len(parts) if parts else None


def main():
    print("Loading observations + var metadata ...")
    df = pd.read_pickle(DATA / "df_analysis_slim.pkl").drop_duplicates(
        ["yearid_id", "question_id"])
    varmeta = pd.read_parquet(
        DATA / "gss_train_vars_corrected_binarized_again.parquet")
    varmeta["n_cat"] = varmeta["code_labels"].apply(count_categories)
    nc_lookup = dict(zip(varmeta["variable_name"], varmeta["n_cat"]))

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

    # Collect (y, pred, n_cat) per variable; drop vars failing class-balance.
    y_all, p_all, nc_all, var_ids = [], [], [], []
    for i, var in enumerate(bare):
        if var not in obs_by_var.groups:
            continue
        n_cat = nc_lookup.get(var)
        if not n_cat or n_cat <= 1:
            continue
        obs_sub = obs_by_var.get_group(var)[["yearid_id", "binarized"]]
        pred_sub = wide[["yearid_id", var]].rename(columns={var: "predicted"})
        m = obs_sub.merge(pred_sub, on="yearid_id", how="inner").dropna(
            subset=["binarized", "predicted"])
        n_pos = int(m["binarized"].sum())
        n_neg = int(len(m) - n_pos)
        if n_pos < 10 or n_neg < 10 or len(m) < 30:
            continue
        y_all.append(m["binarized"].to_numpy())
        p_all.append(m["predicted"].to_numpy())
        nc_all.append(np.full(len(m), int(n_cat), dtype=np.int32))
        var_ids.append(var)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(bare)} vars")

    y = np.concatenate(y_all)
    p = np.concatenate(p_all)
    nc = np.concatenate(nc_all)
    # Per-variable n_cat for counting N questions.
    var_nc = pd.Series([nc_lookup[v] for v in var_ids], index=var_ids).astype(int)

    print(f"Pooled obs: {len(y):,} from {len(var_ids):,} vars")

    rows = []
    for n_cat in sorted(np.unique(nc)):
        mask = nc == n_cat
        yy, pp = y[mask], p[mask]
        if yy.sum() < 10 or (len(yy) - yy.sum()) < 10:
            continue
        auc = float(roc_auc_score(yy, pp))
        n_vars = int((var_nc == n_cat).sum())
        rows.append({
            "n_category": int(n_cat),
            "parity": "even" if n_cat % 2 == 0 else "odd",
            "auc": auc,
            "n_vars": n_vars,
            "n_obs": int(len(yy)),
        })

    out = OUT / "response_category_auc.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"Saved: {out} ({len(rows)} rows)")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
