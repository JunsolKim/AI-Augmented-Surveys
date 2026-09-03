from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DEMO_VARS = ["age_c", "cohort_c", "sex", "race_c", "income_c", "degree", "relig_c"]

# Mapping: (demographic_var, substring_in_value) -> synthetic question text
QUESTION_MAP = {
    # Age
    ("age_c", "18"):    "Are you 18 years or older but below 25 years old?",
    ("age_c", "25"):    "Are you 25 years or older but below 35 years old?",
    ("age_c", "35"):    "Are you 35 years or older but below 45 years old?",
    ("age_c", "45"):    "Are you 45 years or older but below 55 years old?",
    ("age_c", "55"):    "Are you 55 years or older but below 65 years old?",
    ("age_c", "65"):    "Are you 65 years or older?",
    # Cohort
    ("cohort_c", "-1927"):     "Were you born in 1927 or earlier?",
    ("cohort_c", "1928"):      "Were you born in 1928 or later but before 1946?",
    ("cohort_c", "1946"):      "Were you born in 1946 or later but before 1965?",
    ("cohort_c", "1965"):      "Were you born in 1965 or later but before 1981?",
    ("cohort_c", "1981"):      "Were you born in 1981 or later but before 2004?",
    # Sex
    ("sex", "male"):    "Are you male?",
    ("sex", "female"):  "Are you female?",
    # Race
    ("race_c", "White"):    "Is your race White?",
    ("race_c", "Black"):    "Is your race Black or African American?",
    ("race_c", "Hispanic"): "Is your race Hispanic?",
    ("race_c", "Asian"):    "Is your race Asian?",
    ("race_c", "Other"):    "Is your race none of the following: White, Black or African American, Hispanic, or Asian?",
    # Income
    ("income_c", "-15000"):        "Is your family's annual income less than $15,000?",
    ("income_c", "15000-25000"):   "Is your family's annual income between $15,000 and $25,000?",
    ("income_c", "25000-35000"):   "Is your family's annual income between $25,000 and $35,000?",
    ("income_c", "35000-45000"):   "Is your family's annual income between $35,000 and $45,000?",
    ("income_c", "45000-55000"):   "Is your family's annual income between $45,000 and $55,000?",
    ("income_c", "55000-"):        "Is your family's annual income more than $55,000?",
    # Degree
    ("degree", "less than high"):       "Have you not graduated from a high school?",
    ("degree", "high school"):          "Have you graduated from a high school but do not hold a degree from Associate/junior college or Bachelor's degree?",
    ("degree", "associate/junior college"): "Do you have a degree from an Associate/Junior college but not a Bachelor's degree?",
    ("degree", "bachelor"):             "Do you have a Bachelor's degree but do not have a graduate degree?",
    ("degree", "graduate"):             "Do you have a graduate degree?",
    # Religion
    ("relig_c", "Protestant or Catholic"): "What is your religious preference? Is it Protestant or Catholic?",
    ("relig_c", "None"):                   "What is your religious preference? Is it no religion?",
    ("relig_c", "Others"):                 "What is your religious preference? Do you have religious preference other than Protestant or Catholic?",
}


def find_question(demo_var: str, value: str) -> str:
    """Look up the synthetic question for a demographic variable and value."""
    for (var, substr), question in QUESTION_MAP.items():
        if var == demo_var and substr in str(value):
            return question
    return f"Is your {demo_var} equal to {value}?"


def main():
    df_analysis = pd.read_pickle(DATA_DIR / "df_analysis.pkl")
    year_order_map = dict(zip(df_analysis.year, df_analysis.year_order))

    # Extract unique demographics per respondent-year
    demographics = (
        df_analysis[["yearid_id", "year"] + DEMO_VARS]
        .drop_duplicates()
        .sort_values(["yearid_id", "year"])
        .reset_index(drop=True)
    )
    demographics.to_parquet(DATA_DIR / "demographics.parquet")

    # Generate synthetic binary questions
    demo_dfs = []
    for var in DEMO_VARS:
        valid = demographics.loc[pd.notnull(demographics[var])].copy()
        unique_values = sorted(valid[var].unique())

        for val in unique_values:
            df_sub = valid[["yearid_id", "year", var]].rename(columns={var: "response"}).copy()
            df_sub["binarized"] = (df_sub["response"] == val).astype(int)
            df_sub["variable"] = f"{var}_{val}"
            df_sub["question"] = find_question(var, val)
            demo_dfs.append(df_sub)

    demo_long = pd.concat(demo_dfs, ignore_index=True)
    demo_long["question_id"] = LabelEncoder().fit_transform(demo_long["question"])
    demo_long["year_order"] = demo_long["year"].map(year_order_map)

    demo_long.to_csv(DATA_DIR / "demographics_long.csv", index=False)

    print(f"Demographics long shape: {demo_long.shape}")
    print(f"# synthetic questions: {demo_long.variable.nunique()}")


if __name__ == "__main__":
    main()
