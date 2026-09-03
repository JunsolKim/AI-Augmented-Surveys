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

# Reuse the per-respondent AUC helper from the existing pipeline.
sys.path.insert(0, str(BASE / "3_analysis-prediction"))
from prep_fig_alpaca_vs_mf import individual_aucs  # noqa: E402


TASKS = [
    ("impute", "Missing Data Imputation"),
    ("partial", "Retrodiction"),
    ("total", "Unasked Opinion Prediction"),
]
MODEL_STEM = "maicomputer_alpaca-native"


# --- demographic bins ------------------------------------------------------


def _region4(r: float) -> str | float:
    if pd.isna(r):
        return np.nan
    r = int(r)
    if r <= 2:
        return "NE"
    if r <= 4:
        return "MW"
    if r <= 7:
        return "S"
    return "W"


def _age_group(a: float) -> str | float:
    if pd.isna(a):
        return np.nan
    a = int(a)
    if 18 <= a <= 29:
        return "18-29"
    if a <= 44:
        return "30-44"
    if a <= 59:
        return "45-59"
    if a <= 74:
        return "60-74"
    if a <= 89:
        return "75-89"
    return np.nan


def _period_group(y: int) -> str:
    y = int(y)
    if y <= 1979:
        return "1970s"
    if y <= 1989:
        return "1980s"
    if y <= 1999:
        return "1990s"
    if y <= 2009:
        return "2000s"
    return "2010s"


DEGREE_MAP = {
    0: "below_hs", 1: "hs", 2: "some_college", 3: "bs", 4: "bs_above",
}
SEX_MAP = {1: "male", 2: "female"}
RACE_MAP = {1: "White", 2: "Black", 3: "Other"}
PARTYID_MAP = {
    0: "strong democrat", 1: "weak democrat", 2: "lean toward democrat",
    3: "independent", 4: "lean toward republican",
    5: "weak republican", 6: "strong republican", 7: "other party",
}


def build_demographics() -> pd.DataFrame:
    gss = pd.read_parquet(
        DATA / "gss_df_merged.parquet",
        columns=["year", "id", "sex", "race", "degree", "region", "age",
                 "partyid", "realinc", "wtssall", "wtssnrps"],
    )
    gss["yearid"] = gss["year"].astype(int) * 10000 + gss["id"].fillna(0).astype(int)

    gss["sex"] = gss["sex"].map(SEX_MAP)
    gss["race"] = gss["race"].map(RACE_MAP)
    gss["degree"] = gss["degree"].map(DEGREE_MAP)
    gss["region4"] = gss["region"].apply(_region4)
    gss["age_group"] = gss["age"].apply(_age_group)
    gss["period_group"] = gss["year"].apply(_period_group)
    gss["birth_cohort"] = gss["year"].astype("Int64") - gss["age"].astype("Int64")
    gss["partyid7"] = gss["partyid"].map(PARTYID_MAP)

    # Income quintiles per year from realinc (inflation-adjusted family income).
    def _qcut(x: pd.Series) -> pd.Series:
        try:
            q = pd.qcut(x, 5, labels=[f"Income Q{i}" for i in range(1, 6)],
                        duplicates="drop")
            return q.astype("object")
        except ValueError:
            return pd.Series([np.nan] * len(x), index=x.index)
    gss["income_q5"] = gss.groupby("year")["realinc"].transform(_qcut)
    gss.loc[gss["income_q5"].isna(), "income_q5"] = "Income NA"

    # 2021 GSS uses wtssnrps.
    gss.loc[gss["year"] == 2021, "wtssall"] = gss.loc[gss["year"] == 2021, "wtssnrps"]

    return gss[["yearid", "year", "sex", "race", "degree", "region4",
                "income_q5", "age", "age_group", "birth_cohort",
                "period_group", "partyid7", "wtssall"]]


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
    for split, task_label in TASKS:
        wide_path = PRED / (
            f"{MODEL_STEM}_{split}_10_128_50__resample1_wide.parquet"
        )
        if not wide_path.exists():
            print(f"  [MISS] {wide_path.name}")
            continue
        print(f"\n=== {task_label} ({split}) ===")
        ia = individual_aucs(wide_path, df_obs)
        merged = (
            ia.merge(yid_to_yearid, on="yearid_id", how="left")
              .merge(demo, on="yearid", how="left")
        )
        merged.insert(0, "task", task_label)
        all_rows.append(merged)

    if not all_rows:
        print("No results produced.")
        return

    final = pd.concat(all_rows, ignore_index=True)
    out_path = OUT / "individualauc_per_respondent.parquet"
    final.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(final):,} rows)  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
