from pathlib import Path
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

np.random.seed(1234)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_gen"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "maicomputer_alpaca-native"
TASKS = [
    ("impute", "Missing Data Imputation"),
    ("partial", "Retrodiction"),
    ("total", "Unasked Opinion Prediction"),
]
CONDITIONS = [  # (label, use_demo, use_poli)
    ("D0P0", False, False),
    ("D0P1", False, True),   # default: no suffix in filename
    ("D1P0", True, False),
    ("D1P1", True, True),
]


def find_pred_file(split: str, use_demo: bool, use_poli: bool) -> Path | None:
    if use_demo is False and use_poli is True:
        pattern = f"{MODEL}_{split}_0_*_resample1_long.parquet"
        files = [f for f in sorted(glob.glob(str(PRED / pattern)))
                 if "_True_" not in f and "_False_" not in f]
    else:
        pattern = f"{MODEL}_{split}_0_*_resample1_{use_demo}_{use_poli}_long.parquet"
        files = sorted(glob.glob(str(PRED / pattern)))
    return Path(files[0]) if files else None


def main() -> None:
    df_obs = (pd.read_pickle(DATA / "df_analysis_slim.pkl")
                .drop_duplicates(["yearid_id", "question_id"]))
    var_year = df_obs.groupby("variable")["year"].nunique()
    multi_qids = set(df_obs.loc[df_obs["variable"].isin(set(var_year[var_year > 1].index)),
                                "question_id"])
    df_obs = df_obs[["yearid_id", "question_id", "binarized"]]
    print(f"multi-year questions: {len(multi_qids):,}")

    rows = []
    for split, task_label in TASKS:
        for cond, use_demo, use_poli in CONDITIONS:
            f = find_pred_file(split, use_demo, use_poli)
            if f is None:
                print(f"  [MISS] {split} {cond}")
                continue
            ld = pd.read_parquet(f)
            v = ld.loc[ld["validation_0"] == 1,
                       ["yearid_id", "question_id", "response_0"]]
            v = v.merge(df_obs, on=["yearid_id", "question_id"], how="inner")
            v = v.dropna(subset=["binarized", "response_0"])
            if split == "partial":
                v = v[v["question_id"].isin(multi_qids)]
            if v["binarized"].nunique() < 2:
                print(f"  [SKIP] {split} {cond}: single-class")
                continue
            auc = float(roc_auc_score(v["binarized"], v["response_0"]))
            rows.append({
                "task": task_label, "split": split, "condition": cond,
                "use_demo": use_demo, "use_poli": use_poli,
                "auc": auc, "n": int(len(v)),
            })
            print(f"  {split:7s} {cond}: AUC={auc:.4f}  n={len(v):,}")

    out = OUT / "demographic_cv.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
