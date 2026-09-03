from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_PATH = BASE / "data" / "fig_table_gen" / "opinionauc_per_varyear.parquet"
OUT_DIR = BASE / "data" / "fig_table_gen"
OUT_PATH = OUT_DIR / "sorting_by_year.parquet"

TASK = "Retrodiction"  # partial split; matches SM figure A12


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PATH)
    sub = (df.loc[df["task"] == TASK,
                  ["variable", "year", "auc", "abs_corr_with_ideo"]]
             .dropna(subset=["auc", "abs_corr_with_ideo", "year"])
             .reset_index(drop=True))
    print(f"retrodiction (variable, year) rows: {len(sub):,}")
    print(f"  year range: {sub['year'].min()}-{sub['year'].max()}")
    print(f"  abs_corr_with_ideo mean={sub['abs_corr_with_ideo'].mean():.3f} "
          f"sd={sub['abs_corr_with_ideo'].std():.3f}")

    sub.to_parquet(OUT_PATH, index=False)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
