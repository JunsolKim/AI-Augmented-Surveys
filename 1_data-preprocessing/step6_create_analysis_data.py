import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from binarization_utils import (
    POLITICAL_QUESTIONS,
    split_political_variables,
    apply_political_questions,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_VARS_FILE = DATA_DIR / "gss_train_vars_corrected_binarized_again.parquet"

# polviews response_int -> binarized for each variant
POLVIEWS_BINARIZATION = {
    "polviews1": {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},  # liberal
    "polviews2": {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1},  # conservative
    "polviews3": {1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0},  # moderate
}

# partyid response_int -> binarized for each variant
PARTYID_BINARIZATION = {
    "partyid_dem": {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},  # democrat
    "partyid_rep": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0},  # republican
    "partyid_ind": {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},  # independent
    "partyid_oth": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1},  # other party
}


def build_binarization_table(train_vars):
    """Build a (variable, response_int) -> binarized mapping from the train vars parquet.

    Parses binarized JSON keys like "1: Very happy" to extract the response code.
    """
    records = {"variable": [], "response_int": [], "binarized": []}
    for _, row in train_vars.iterrows():
        binarized_dict = json.loads(row["binarized"])
        for key, val in binarized_dict.items():
            code_str = key.split(":")[0].strip()
            try:
                code = int(code_str)
            except ValueError:
                continue
            records["variable"].append(row["variable_name"].lower())
            records["response_int"].append(code)
            records["binarized"].append(val)
    return pd.DataFrame(records)


def add_partyid_rows(df):
    """Add partyid rows from gss_df_merged since partyid is not in gss_qna_all.

    Reads partyid from the wide-format source and creates long-format rows
    matching the structure of df.
    """
    gss_wide = pd.read_parquet(DATA_DIR / "gss_df_merged.parquet", columns=["year", "id", "partyid"])
    gss_wide = gss_wide.loc[pd.notnull(gss_wide["partyid"])].copy()
    gss_wide["partyid"] = gss_wide["partyid"].astype(int)
    gss_wide["yearid"] = gss_wide["year"].astype(int) * 100000 + gss_wide["id"].astype(int)

    # Get a template row from df for column alignment
    template_cols = df.columns.tolist()

    # Build long-format partyid rows
    rows = pd.DataFrame({
        "yearid_year": (gss_wide["yearid"].astype(str) + gss_wide["year"].astype(str)).astype(int),
        "year": gss_wide["year"].astype(int),
        "variable": "partyid",
        "response_int": gss_wide["partyid"],
        "question": "Generally speaking, do you usually think of yourself as a Republican, Democrat, Independent, or what?",
    })

    # Carry over yearid_year from df to get consistent yearid mapping
    existing_yearid = df[["yearid_year", "year"]].drop_duplicates()
    rows = rows.merge(existing_yearid[["yearid_year"]], on="yearid_year", how="inner")

    # Add missing columns as NaN
    for col in template_cols:
        if col not in rows.columns:
            rows[col] = pd.NA

    return pd.concat([df, rows[template_cols]], ignore_index=True)


def main():
    # Load data
    df = pd.read_parquet(DATA_DIR / "gss_qna_all.parquet")
    rename_dict = {f"demo_{v}": v for v in ["age_c", "cohort_c", "sex", "race_c", "income_c", "degree", "relig_c", "polviews"]}
    df = df.rename(columns=rename_dict)

    # Load train vars: only use_variable_final == 1
    train_vars = pd.read_parquet(TRAIN_VARS_FILE)
    train_vars = train_vars[train_vars["use_variable_final"] == 1].copy()

    # Filter to only variables in train vars (lowercase to match data columns)
    # Also keep polviews (needed for political variable splitting)
    valid_variables = {v.lower() for v in train_vars["variable_name"]}
    df = df[df["variable"].isin(valid_variables | {"polviews"})].copy()

    # Add partyid rows from wide-format source
    df = add_partyid_rows(df)

    # Split polviews -> polviews1/2/3 and partyid -> partyid_dem/rep/ind/oth
    df = split_political_variables(df)
    df = apply_political_questions(df)

    # Binarize responses
    binarization_table = build_binarization_table(train_vars)

    # Add political variable binarization entries
    poli_records = {"variable": [], "response_int": [], "binarized": []}
    for var_map in [POLVIEWS_BINARIZATION, PARTYID_BINARIZATION]:
        for var, mapping in var_map.items():
            for resp, bval in mapping.items():
                poli_records["variable"].append(var)
                poli_records["response_int"].append(resp)
                poli_records["binarized"].append(bval)
    binarization_table = pd.concat([binarization_table, pd.DataFrame(poli_records)], ignore_index=True)

    df = df.merge(binarization_table, on=["variable", "response_int"], how="left")
    df = df[pd.notnull(df["binarized"])].copy()

    # Encode IDs
    df["yearid_id"] = LabelEncoder().fit_transform(df["yearid_year"])
    df["question_id"] = LabelEncoder().fit_transform(df["question"])
    df["year_order"] = LabelEncoder().fit_transform(df["year"])

    # Save yearly wide-format pivots
    for year in sorted(df.year.unique()):
        df_wide = df.loc[df.year == year, ["yearid_id", "variable", "binarized"]].pivot(
            index="yearid_id", columns="variable", values="binarized"
        )
        df_wide.to_parquet(DATA_DIR / f"df_wide_{year}.parquet")

    # Save main analysis file
    df.to_pickle(DATA_DIR / "df_analysis.pkl")

    print(f"Analysis data shape: {df.shape}")
    print(f"# variables: {df.variable.nunique()}")
    print(f"# respondents: {df.yearid_id.nunique()}")
    print(f"Year range: {df.year.min()}-{df.year.max()}")


if __name__ == "__main__":
    main()
