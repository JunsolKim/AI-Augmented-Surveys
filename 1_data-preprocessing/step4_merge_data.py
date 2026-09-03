from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    gss_df = pd.read_stata(DATA_DIR / "gss7221_r2.dta", convert_categoricals=False)
    gss_df = gss_df.loc[gss_df.year.astype(int) <= 2021].copy()  # keep up to 2021
    gss_df["yearid"] = gss_df["year"].astype(str) + gss_df["id"].astype(str).str.zfill(4)
    gss_df["yearid"] = gss_df["yearid"].astype(int)
    gss_df = gss_df.sort_values(["yearid", "year"]).reset_index(drop=True)
    gss_df.to_parquet(DATA_DIR / "gss_df_merged.parquet")

    print(f"# of variables: {len(gss_df.columns)}")
    print(f"# of respondents: {gss_df['yearid'].nunique()}")
    print(f"# of years: {gss_df['year'].nunique()}")


if __name__ == "__main__":
    main()
