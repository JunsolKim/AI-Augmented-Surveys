import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_VARS_FILE = DATA_DIR / "gss_train_vars_corrected_binarized_again.parquet"

DEMOGRAPHIC_VARS = ["yearid", "year", "age_c", "cohort_c", "sex", "race_c", "income_c", "degree", "relig_c"]
DEMO_COLS = ["age_c", "cohort_c", "sex", "race_c", "income_c", "degree", "relig_c"]
NETWORK_VARS = ["r_memnum", "n_size", "n_friend", "r_contact",
                "r_inperson", "r_byphone", "r_letters", "r_meetings", "r_byemail"]

# Asian ethnic codes
ASIAN_CODES = [5, 16, 20, 31, 37, 40, 301, 302, 304, 306, 307, 401, 402, 403, 404, 405, 406, 408, 499]
# Hispanic ethnic codes
HISPANIC_CODES = [17, 22, 25, 28, 38, 501, 504, 505, 506, 509, 510, 513, 602, 603, 605, 606, 607, 801, 803]

# Contact frequency midpoint mapping (GSS ordinal -> approximate count)
CONTACT_MIDPOINTS = {1: 1, 2: 1.5, 3: 4, 4: 8, 5: 13, 6: 20.5, 7: 37.5, 8: 50}


def load_dicts():
    """Load metadata dictionaries from Step 2."""
    dicts = {}
    for name in ["var_question_dict", "online_value_label_dict", "module_dict", "var_to_subject_dict"]:
        with open(DATA_DIR / f"{name}.pkl", "rb") as f:
            dicts[name] = pickle.load(f)
    return dicts


def supplement_dicts_from_train_vars(var_question_dict, online_value_label_dict):
    """Supplement metadata dicts with data from the train vars parquet.

    Adds question text and value labels for variables that exist in the train
    vars file but are missing from the Step 2 dictionaries.
    """
    train_vars = pd.read_parquet(TRAIN_VARS_FILE)
    train_vars = train_vars[train_vars["use_variable_final"] == 1].copy()

    for _, row in train_vars.iterrows():
        vname = row["variable_name"].lower()

        # Supplement question text
        if vname not in var_question_dict:
            question = row.get("processed_question") or row.get("question")
            if pd.notna(question):
                var_question_dict[vname] = question

        # Supplement value labels from binarized keys (e.g. "1: Very happy" -> {1: "very happy"})
        if vname not in online_value_label_dict and pd.notna(row.get("binarized")):
            binarized_dict = json.loads(row["binarized"])
            labels = {}
            for key in binarized_dict:
                parts = key.split(":", 1)
                if len(parts) == 2:
                    try:
                        code = int(parts[0].strip())
                        label = parts[1].strip().lower()
                        labels[str(code)] = label
                    except ValueError:
                        continue
            if labels:
                online_value_label_dict[vname] = labels


def get_train_var_names():
    """Get the set of variable names (lowercased) with use_variable_final == 1."""
    train_vars = pd.read_parquet(TRAIN_VARS_FILE)
    train_vars = train_vars[train_vars["use_variable_final"] == 1]
    return {v.lower() for v in train_vars["variable_name"]}


def create_network_variables(df, var_question_dict, module_dict, var_to_subject_dict, online_value_label_dict):
    """Create derived network/social capital variables."""
    # Voluntary associational memberships (direct copy)
    df["r_memnum"] = df["memnum"]
    # Network size from "important matters" question
    df["n_size"] = df["numgiven"]

    for derived, source in [("r_memnum", "memnum"), ("n_size", "numgiven")]:
        var_question_dict[derived] = var_question_dict[source]
        module_dict[derived] = module_dict[source]
        var_to_subject_dict[derived] = var_to_subject_dict[source]

    # Friend network size (binned)
    df["d_friend"] = (df["friends"] == 1).astype(float)
    df.loc[pd.isnull(df.friends), "d_friend"] = np.nan
    df["n_friend"] = np.nan
    df.loc[df.d_friend == 0, "n_friend"] = 0
    df.loc[df.numfrend <= 2, "n_friend"] = 0
    df.loc[(df.numfrend > 2) & (df.numfrend <= 4), "n_friend"] = 1
    df.loc[(df.numfrend > 4) & (df.numfrend <= 8), "n_friend"] = 2
    df.loc[(df.numfrend > 8) & (df.numfrend != 94), "n_friend"] = 3
    for d in [var_question_dict, module_dict, var_to_subject_dict]:
        d["n_friend"] = d["numfrend"]

    # Contact frequency (binned)
    df["r_contact"] = np.nan
    df.loc[df.numcntct <= 6, "r_contact"] = 0
    df.loc[(df.numcntct > 6) & (df.numcntct <= 15), "r_contact"] = 1
    df.loc[(df.numcntct > 15) & (df.numcntct <= 30), "r_contact"] = 2
    df.loc[(df.numcntct > 30) & (df.numcntct < 900), "r_contact"] = 3
    for d in [var_question_dict, module_dict, var_to_subject_dict]:
        d["r_contact"] = d["numcntct"]

    # Contact methods (mapped to midpoint counts)
    for method in ["inperson", "byphone", "letters", "meetings", "byemail"]:
        rvar = f"r_{method}"
        df[rvar] = df[method].map(CONTACT_MIDPOINTS)
        var_question_dict[rvar] = var_question_dict[method]
        module_dict[rvar] = module_dict[method]
        var_to_subject_dict[rvar] = var_to_subject_dict[method]

    # Value labels for derived variables
    online_value_label_dict["n_friend"] = {0: "x<=2", 1: "2<x<=4", 2: "4<x<=8", 3: "8<x"}
    online_value_label_dict["r_contact"] = {0: "x<=6", 1: "6<x<=15", 2: "15<x<=30", 3: "30<x"}
    for method in ["r_inperson", "r_byphone", "r_letters", "r_meetings", "r_byemail"]:
        online_value_label_dict[method] = {k: str(v) for k, v in CONTACT_MIDPOINTS.items()}

    return df


def create_demographic_bins(df, var_question_dict, module_dict, var_to_subject_dict, online_value_label_dict):
    """Create binned demographic variables."""
    # Age groups
    df["age_c"] = np.nan
    age_bins = [(18, 24, 1), (25, 34, 2), (35, 44, 3), (45, 54, 4), (55, 64, 5), (65, 90, 6)]
    for lo, hi, val in age_bins:
        df.loc[(df["age"] >= lo) & (df["age"] <= hi), "age_c"] = val
    online_value_label_dict["age_c"] = {1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65-"}

    # Birth cohort
    df["cohort_c"] = np.nan
    cohort_bins = [(1883, 1927, 1), (1928, 1945, 2), (1946, 1964, 3), (1965, 1980, 4), (1981, 2012, 5)]
    for lo, hi, val in cohort_bins:
        df.loc[(df["cohort"] >= lo) & (df["cohort"] <= hi), "cohort_c"] = val
    online_value_label_dict["cohort_c"] = {1: "-1927", 2: "1928-1945", 3: "1946-1964", 4: "1965-1980", 5: "1981-2003"}

    # Race/ethnicity (refined with ethnic codes)
    df["race_c"] = df["race"].copy()
    df.loc[df["race"] == 3, "race_c"] = 5  # "Other" initially for race=3
    df.loc[(df["race_c"] == 5) & (df["ethnic"].isin(ASIAN_CODES)), "race_c"] = 3    # Asian
    df.loc[(df["race_c"] == 5) & (df["ethnic"].isin(HISPANIC_CODES)), "race_c"] = 4  # Hispanic
    online_value_label_dict["race_c"] = {1: "White", 2: "Black", 3: "Asian", 4: "Hispanic", 5: "Other"}

    # Real income brackets
    df["income_c"] = np.nan
    income_bins = [(0, 15000, 1), (15000, 25000, 2), (25000, 35000, 3),
                   (35000, 45000, 4), (45000, 55000, 5), (55000, 200000, 6)]
    for lo, hi, val in income_bins:
        df.loc[(df["realinc"] >= lo) & (df["realinc"] < hi), "income_c"] = val
    online_value_label_dict["income_c"] = {
        1: "-15000", 2: "15000-25000", 3: "25000-35000",
        4: "35000-45000", 5: "45000-55000", 6: "55000-",
    }

    # Religion
    df["relig_c"] = np.nan
    df.loc[(df["relig"] == 1) | (df["relig"] == 2), "relig_c"] = 1  # Protestant/Catholic
    df.loc[(df["relig_c"] != 1) & (df["relig"] != 4), "relig_c"] = 2  # Others
    df.loc[df["relig"] == 4, "relig_c"] = 3  # None
    online_value_label_dict["relig_c"] = {1: "Protestant or Catholic", 2: "Others", 3: "None"}

    # Copy metadata from source variables
    for derived, source in [("age_c", "age"), ("cohort_c", "cohort"), ("race_c", "race"),
                            ("income_c", "income"), ("relig_c", "relig")]:
        var_question_dict[derived] = var_question_dict[source]
        module_dict[derived] = module_dict[source]
        var_to_subject_dict[derived] = var_to_subject_dict[source]

    return df


def create_demo_labels(df, online_value_label_dict):
    """Create text-label columns (demo_*) for each demographic variable."""
    include_vars = ["age_c", "cohort_c", "sex", "race_c", "income_c", "degree", "relig_c", "polviews"]
    str_key_vars = {"sex", "degree", "polviews"}  # variables whose dict keys are strings of ints

    for var in include_vars:
        labels = online_value_label_dict[var]
        mask = pd.notnull(df[var])
        if var in str_key_vars:
            df.loc[mask, f"demo_{var}"] = df.loc[mask, var].apply(
                lambda x: labels.get(str(int(x)))
            )
        else:
            df.loc[mask, f"demo_{var}"] = df.loc[mask, var].apply(
                lambda x: labels.get(x)
            )
    return df


def resolve_response_text(variable, response_int, online_value_label_dict):
    """Map (variable, response_int) to response text."""
    try:
        if variable in ("age_c", "cohort_c", "race_c", "income_c", "relig_c"):
            return online_value_label_dict[variable][int(response_int)]
        elif variable in ("n_size", "r_memnum"):
            return str(response_int)
        elif variable in NETWORK_VARS:
            return online_value_label_dict[variable][int(response_int)]
        else:
            return online_value_label_dict[variable][str(response_int)]
    except KeyError:
        return None


def main():
    dicts = load_dicts()
    var_question_dict = dicts["var_question_dict"]
    online_value_label_dict = dicts["online_value_label_dict"]
    module_dict = dicts["module_dict"]
    var_to_subject_dict = dicts["var_to_subject_dict"]

    df = pd.read_parquet(DATA_DIR / "gss_df_merged.parquet")

    # Create derived variables
    df = create_network_variables(df, var_question_dict, module_dict, var_to_subject_dict, online_value_label_dict)
    df = create_demographic_bins(df, var_question_dict, module_dict, var_to_subject_dict, online_value_label_dict)
    df = create_demo_labels(df, online_value_label_dict)

    # Supplement dicts with train vars data (fills gaps from Step 2 filtering)
    supplement_dicts_from_train_vars(var_question_dict, online_value_label_dict)

    # Load train var names for variable selection
    train_var_names = get_train_var_names()

    # Filter valid rows and create composite keys
    df = df.loc[pd.notnull(df.year) & pd.notnull(df["id"])].copy()
    df["yearid"] = df["yearid"].astype(int)
    df["year"] = df["year"].astype(int)
    df["yearid_year"] = (df["yearid"].astype(str) + df["year"].astype(str)).astype(int)

    # Determine which columns to melt into long format
    include_demo = {"age_c", "cohort_c", "sex", "race_c", "income_c", "degree", "relig_c", "polviews"}
    valid_vars = sorted(
        (set(df.columns) & train_var_names & set(var_question_dict))
        | include_demo
        | set(NETWORK_VARS)
    )

    not_in_data = train_var_names - set(df.columns)
    if not_in_data:
        print(f"Warning: {len(not_in_data)} train vars not found in data columns: {sorted(not_in_data)[:10]}...")

    demo_label_cols = [f"demo_{v}" for v in ["age_c", "cohort_c", "sex", "race_c", "income_c", "degree", "relig_c", "polviews"]]
    meta_cols = ["yearid", "year", "yearid_year"] + demo_label_cols

    # Build long-format DataFrame
    df_temp_list = []
    for col in tqdm(valid_vars, desc="Building long format"):
        chunk = df[meta_cols].copy()
        chunk["question"] = var_question_dict[col]
        if col in module_dict:
            chunk["module"] = module_dict[col]
        if col in var_to_subject_dict:
            chunk["subject"] = [var_to_subject_dict[col]] * len(chunk)
        chunk["variable"] = col
        chunk["response_int"] = df[col]
        chunk = chunk.loc[pd.notnull(chunk.response_int)].copy()
        chunk["response_int"] = chunk["response_int"].astype(int)
        df_temp_list.append(chunk)

    long_df = pd.concat(df_temp_list, axis=0, ignore_index=True)

    # Demographics completeness flag
    long_df["demographics_notnull"] = 1
    for var in DEMO_COLS:
        long_df.loc[pd.isnull(long_df[f"demo_{var}"]), "demographics_notnull"] = 0

    # Resolve response text
    non_count = 0
    responses = []
    for v, r in tqdm(zip(long_df["variable"], long_df["response_int"]), total=len(long_df), desc="Resolving responses"):
        text = resolve_response_text(v, r, online_value_label_dict)
        if text is None:
            non_count += 1
        responses.append(text)
    long_df["response"] = responses

    print(f"Responses not in value label dict: {non_count}")

    long_df = long_df.sort_values(["yearid", "year", "variable"]).reset_index(drop=True)
    long_df.to_parquet(DATA_DIR / "gss_qna_all.parquet")

    print(f"Output shape: {long_df.shape}")
    print(f"Columns: {list(long_df.columns)}")


if __name__ == "__main__":
    main()
