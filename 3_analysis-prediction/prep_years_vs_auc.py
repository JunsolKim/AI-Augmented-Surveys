import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT_DIR = DATA / "fig_table_gen"
MODEL = "maicomputer_alpaca-native"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(1234)

    df = (pd.read_pickle(DATA / "df_analysis_slim.pkl")
            .drop_duplicates(["yearid_id", "question_id"]))

    var_n_year = (df.groupby("variable")["year"].nunique()
                    .reset_index(name="n_year"))
    var_n_year = var_n_year[var_n_year["n_year"] >= 2]
    var_qid = df[["variable", "question_id"]].drop_duplicates()
    var_info = var_n_year.merge(var_qid, on="variable")
    print(f"multi-year variables: {var_info['variable'].nunique():,}")

    files = sorted(glob.glob(str(
        PRED / f"{MODEL}_partial_0_*_resample1_long.parquet")))
    assert files, "partial fold-0 long parquet not found"
    long_df = pd.read_parquet(files[0])

    val = (long_df.loc[long_df["validation_0"] == 1,
                       ["yearid_id", "question_id", "response_0"]]
                  .merge(df[["yearid_id", "question_id", "binarized"]],
                         on=["yearid_id", "question_id"], how="inner")
                  .merge(var_info[["question_id", "n_year", "variable"]],
                         on="question_id", how="inner")
                  .dropna(subset=["binarized", "response_0"]))
    print(f"validation obs: {len(val):,}")

    rows = []
    for ny, g in val.groupby("n_year"):
        if g["binarized"].nunique() < 2:
            continue
        auc = float(roc_auc_score(g["binarized"], g["response_0"]))
        rows.append({"n_year": int(ny), "auc": auc,
                     "n_obs": int(len(g)),
                     "n_vars": int(g["variable"].nunique())})
        print(f"  n_year={ny:>2d}: auc={auc:.4f}  "
              f"n={len(g):>7,}  n_vars={g['variable'].nunique():>4}")

    out = pd.DataFrame(rows).sort_values("n_year").reset_index(drop=True)
    out_path = OUT_DIR / "years_vs_auc.parquet"
    out.to_parquet(out_path, index=False)
    print(f"saved: {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
