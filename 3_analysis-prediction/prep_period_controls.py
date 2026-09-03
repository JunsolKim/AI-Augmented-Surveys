from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT_PATH = DATA / "fig_table_gen" / "_period_controls.parquet"


def main() -> None:
    print("Loading observations ...")
    # Same de-duplication the individual-AUC preps use, so yearid_id and the
    # answered-item set line up exactly with the AUC frames these join to.
    df = (
        pd.read_pickle(DATA / "df_analysis_slim.pkl")
        .drop_duplicates(["yearid_id", "question_id"])
    )
    obs = df[["yearid_id", "variable", "binarized"]].dropna(subset=["binarized"])
    print(f"  obs rows: {len(obs):,}")

    # Item marginals, pooled over every respondent who answered the item.
    p_v = obs.groupby("variable")["binarized"].mean()
    obs = obs.assign(abs_diff=(obs["variable"].map(p_v) - 0.5).abs())

    ctrl = (
        obs.groupby("yearid_id")
        .agg(n_q=("variable", "size"), mean_diff=("abs_diff", "mean"))
        .reset_index()
    )
    ctrl["yearid_id"] = ctrl["yearid_id"].astype("int32")
    ctrl["n_q"] = ctrl["n_q"].astype("int64")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ctrl.to_parquet(OUT_PATH, index=False)
    print(f"saved: {OUT_PATH}  ({len(ctrl):,} respondents)")


if __name__ == "__main__":
    main()
