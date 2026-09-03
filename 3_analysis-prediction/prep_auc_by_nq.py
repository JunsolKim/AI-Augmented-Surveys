from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT_PATH = DATA / "fig_table_gen" / "auc_by_nq.parquet"

MODEL = "maicomputer_alpaca-native"
STEM = f"{MODEL}_impute_0_10_128_50__resample1"
# Must match A15_MAX_QS in 2_model-finetuning/Snakefile.
MAX_QS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]
# Per the manuscript caption for Figure A18.
YEARS = [2016, 2018, 2021]


def _auc_for(path: Path, obs: pd.DataFrame) -> tuple[float, int] | None:
    if not path.exists():
        print(f"  [MISS] {path.name}")
        return None
    long_df = pd.read_parquet(path, columns=["yearid_id", "question_id",
                                             "response_0", "validation_0"])
    val = (
        long_df.loc[long_df["validation_0"] == 1,
                    ["yearid_id", "question_id", "response_0"]]
        .merge(obs, on=["yearid_id", "question_id"], how="inner")
        .dropna(subset=["binarized", "response_0"])
    )
    if val["binarized"].nunique() < 2:
        return None
    return float(roc_auc_score(val["binarized"], val["response_0"])), len(val)


def main() -> None:
    print("Loading observations ...")
    df = (
        pd.read_pickle(DATA / "df_analysis_slim.pkl")
        .drop_duplicates(["yearid_id", "question_id"])
    )
    obs = df.loc[df["year"].isin(YEARS),
                 ["yearid_id", "question_id", "binarized"]]
    print(f"  obs rows: {len(obs):,} (years {YEARS})")

    rows = []
    got = _auc_for(PRED / f"{STEM}_long.parquet", obs)
    if got:
        rows.append({"max_q": "all", "auc": got[0], "n": got[1]})
        print(f"  max_q=all: auc={got[0]:.6f}  n={got[1]:,}")

    for nq in MAX_QS:
        got = _auc_for(PRED / f"{STEM}_nq{nq}_long.parquet", obs)
        if got:
            rows.append({"max_q": str(nq), "auc": got[0], "n": got[1]})
            print(f"  max_q={nq:>3}: auc={got[0]:.6f}  n={got[1]:,}")

    out = pd.DataFrame(rows)
    out["n"] = out["n"].astype("int64")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"saved: {OUT_PATH}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
