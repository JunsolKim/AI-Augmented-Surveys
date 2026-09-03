from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PRED = BASE / "data" / "predictions"
OUT = BASE / "data" / "fig_table_gen"

MODEL_PREFIX = "maicomputer_alpaca-native"

SCENARIOS = [
    ("impute",  "Imputation"),
    ("partial", "Retrodiction"),
    ("total",   "Unasked prediction"),
]

THRESHOLDS_PCT = list(range(1, 11))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for skey, slabel in SCENARIOS:
        path = PRED / (f"{MODEL_PREFIX}_{skey}_10_128_50"
                       f"__resample1_varyear.parquet")
        df = pd.read_parquet(path, columns=["GLM_predicted", "obs_wavg"])
        df = df.dropna(subset=["GLM_predicted", "obs_wavg"])
        diff = (df["GLM_predicted"] - df["obs_wavg"]).abs()
        n = int(len(diff))
        print(f"{slabel:20s}: n={n:,}")

        for t_pct in THRESHOLDS_PCT:
            pct = float((diff < t_pct / 100).mean() * 100)
            rows.append({
                "scenario": skey, "scenario_label": slabel,
                "threshold_pct": t_pct,
                "pct_within": pct,
                "n": n,
            })
            print(f"    within {t_pct:>2d}%: {pct:5.1f}%")

    out_df = pd.DataFrame(rows)
    out_path = OUT / "accuracythresholds_alpaca.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
