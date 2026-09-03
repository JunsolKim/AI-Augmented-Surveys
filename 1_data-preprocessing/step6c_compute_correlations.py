from pathlib import Path

import fire
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main(target="marhomo1"):
    df = pd.read_pickle(DATA_DIR / "df_analysis_slim.pkl")
    df = df.drop_duplicates(["yearid_id", "question_id"])

    wide = df.pivot(index="yearid_id", columns="variable", values="binarized")

    if target not in wide.columns:
        raise ValueError(f"Target variable '{target}' not found in df_analysis")

    corr = wide.corrwith(wide[target]).drop(target).dropna().abs().sort_values(ascending=False)
    out = corr.reset_index()
    out.columns = ["variable", "abs_corr"]

    out_path = DATA_DIR / f"corr_with_{target}.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Target: {target}")
    print(f"Saved {len(out)} variables -> {out_path.name}")
    print(f"\nTop 10:")
    for i, row in out.head(10).iterrows():
        print(f"  {i+1:3d}. {row['variable']:<30s} r={row['abs_corr']:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
