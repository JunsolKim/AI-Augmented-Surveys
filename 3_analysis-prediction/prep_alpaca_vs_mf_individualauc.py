import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


np.random.seed(1234)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_gen"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse the per-respondent AUC helper + demographics builder from the
# existing pipeline.
sys.path.insert(0, str(BASE / "3_analysis-prediction"))
sys.path.insert(0, str(BASE / "3_analysis-prediction"))
from prep_fig_alpaca_vs_mf import individual_aucs  # noqa: E402
from prep_individualauc import build_demographics  # noqa: E402


TASKS = [
    ("impute", "Missing Data Imputation"),
    ("partial", "Retrodiction"),
]
MODELS = [
    ("Alpaca-7b", "maicomputer_alpaca-native"),
    ("MF", "mf"),
]


def main() -> None:
    t0 = time.time()

    print("Loading observations ...")
    df = (
        pd.read_pickle(DATA / "df_analysis_slim.pkl")
        .drop_duplicates(["yearid_id", "question_id"])
    )
    df_obs = (
        df[["yearid_id", "yearid", "variable", "binarized"]]
        .dropna(subset=["binarized"])
    )
    yid_to_yearid = df_obs.drop_duplicates("yearid_id")[["yearid_id", "yearid"]]
    print(f"  obs rows: {len(df_obs):,}")

    print("Building demographics (original R bins) ...")
    demo = build_demographics()
    print(f"  demo rows: {len(demo):,}")

    all_rows = []
    for model_label, stem in MODELS:
        for split, task_label in TASKS:
            wide_path = PRED / f"{stem}_{split}_10_128_50__resample1_wide.parquet"
            if not wide_path.exists():
                print(f"  [MISS] {wide_path.name}")
                continue
            print(f"\n=== {model_label} | {task_label} ({split}) ===")
            ia = individual_aucs(wide_path, df_obs)
            merged = (
                ia.merge(yid_to_yearid, on="yearid_id", how="left")
                  .merge(demo, on="yearid", how="left")
            )
            merged.insert(0, "task", task_label)
            merged.insert(1, "model", model_label)
            all_rows.append(merged)

    if not all_rows:
        print("No results produced.")
        return

    final = pd.concat(all_rows, ignore_index=True)
    out_path = OUT / "alpaca_vs_mf_individualauc_per_respondent.parquet"
    final.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(final):,} rows)  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
