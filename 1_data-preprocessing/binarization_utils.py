import ast
import pickle
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Words indicating non-substantive survey instructions
QUESTION_STOPWORDS = [
    "GO", "REMARKS", "EACH", "INTERVIEWER", "PERSON", "ABOVE", "IF",
    "CIRCLE", "PROBE", "INSTRUCTION", "CARD", "RECORD", "NAME",
    "ASK", "ONLY", "YES", "CODE", "Appendix",
]

# Subject categories to exclude entirely
EXCLUDED_SUBJECTS = [
    "Date Of Interview", "Interview", "Sample",
    "Size Of Place Of Interview", "Social Networks", "Surveys",
]

# Political response labels
POLVIEWS_RESPONSES = [
    "moderate, middle of the road", "slightly conservative",
    "conservative", "liberal", "extremely conservative",
    "slightly liberal", "extremely liberal",
]
PARTYID_RESPONSES = [
    "independent, close to democrat", "not very strong democrat",
    "independent (neither, no response)", "strong democrat",
    "not very strong republican", "independent, close to republican",
    "strong republican", "other party",
]

# Political question text
POLITICAL_QUESTIONS = {
    "polviews1": "We hear a lot of talk these days about liberals and conservatives. Are you a liberal?",
    "polviews2": "We hear a lot of talk these days about liberals and conservatives. Are you a conservative?",
    "polviews3": "We hear a lot of talk these days about liberals and conservatives. Are you a moderate or in the middle of the road?",
    "partyid_dem": "Generally speaking, do you usually think of yourself as a Democrat?",
    "partyid_rep": "Generally speaking, do you usually think of yourself as a Republican?",
    "partyid_ind": "Generally speaking, do you usually think of yourself as an Independent?",
    "partyid_oth": "Generally speaking, do you usually think of yourself as an other party?",
}


def load_metadata_dicts():
    """Load all metadata dictionaries from Step 2."""
    names = ["var_question_dict", "online_value_label_dict", "subject_to_var_dict", "var_description_dict"]
    dicts = {}
    for name in names:
        with open(DATA_DIR / f"{name}.pkl", "rb") as f:
            dicts[name] = pickle.load(f)
    return dicts


def get_excluded_variables(var_question_dict, subject_to_var_dict):
    """Build set of variable names to exclude based on stopwords and subject categories."""
    excluded = set()
    for var, question in var_question_dict.items():
        if any(w in question for w in QUESTION_STOPWORDS):
            excluded.add(var)
    for subj in EXCLUDED_SUBJECTS:
        if subj in subject_to_var_dict:
            excluded.update(subject_to_var_dict[subj])
    return excluded


def filter_and_rename(df):
    """Rename demo_ columns to their base names."""
    rename_dict = {f"demo_{v}": v for v in ["age_c", "cohort_c", "sex", "race_c", "income_c", "degree", "relig_c", "polviews"]}
    return df.rename(columns=rename_dict)


def split_political_variables(df):
    """Split polviews and partyid into multiple binary-question variants.

    Returns df with original polviews/partyid removed and variants
    (polviews1-3, partyid_dem/rep/ind/oth) added.
    """
    df_nopol = df.loc[(df.variable != "polviews") & (df.variable != "partyid")]

    variant_names = {
        "polviews": ["polviews1", "polviews2", "polviews3"],
        "partyid": ["partyid_dem", "partyid_rep", "partyid_ind", "partyid_oth"],
    }

    variants = []
    for base_var, names in variant_names.items():
        df_base = df.loc[df.variable == base_var]
        for name in names:
            variant = df_base.copy()
            variant["variable"] = name
            variants.append(variant)

    return pd.concat([df_nopol] + variants, ignore_index=True)


def build_binarization_table(binarization_xlsx_path):
    """Build a (variable, response) -> binarized mapping from the Excel file, plus political variables.

    Returns a DataFrame with columns: [variable, response, binarized].
    """
    response_binarized = pd.read_excel(binarization_xlsx_path)
    raw_data = pd.read_excel(binarization_xlsx_path, sheet_name="raw_data")

    merged = raw_data[["variable", "response_all"]].merge(
        response_binarized, left_on="response_all", right_on="response", how="left"
    )[["variable", "response_all", "binarized", "binarized_corrected", "nominal/check_again"]].drop_duplicates()

    # Apply corrections
    mask_corrected = pd.notnull(merged.binarized_corrected)
    merged.loc[mask_corrected, "binarized"] = merged.loc[mask_corrected, "binarized_corrected"]

    # Filter valid entries
    merged = merged.loc[pd.notnull(merged.binarized) & (merged["nominal/check_again"] != 1)]

    # Expand list-formatted cells into rows
    records = {"variable": [], "response": [], "binarized": []}
    for _, row in merged.iterrows():
        try:
            responses = ast.literal_eval(row["response_all"])
            binarized = ast.literal_eval(row["binarized"])
            records["variable"].extend([row["variable"]] * len(responses))
            records["response"].extend(responses)
            records["binarized"].extend(binarized)
        except (TypeError, ValueError):
            pass

    # Add political variable binarization
    r_pol, r_party = POLVIEWS_RESPONSES, PARTYID_RESPONSES

    pol_binarized = {
        "polviews1": [int("liberal" in r) for r in r_pol],
        "polviews2": [int("conservative" in r) for r in r_pol],
        "polviews3": [int("moderate" in r) for r in r_pol],
    }
    party_binarized = {
        "partyid_dem": [int("democrat" in r) for r in r_party],
        "partyid_rep": [int("republican" in r) for r in r_party],
        "partyid_ind": [int("independent" in r) for r in r_party],
        "partyid_oth": [int("other party" in r) for r in r_party],
    }

    for var, bvals in {**pol_binarized, **party_binarized}.items():
        resps = r_pol if var.startswith("polviews") else r_party
        records["variable"].extend([var] * len(resps))
        records["response"].extend(resps)
        records["binarized"].extend(bvals)

    return pd.DataFrame(records)


def apply_political_questions(df):
    """Set question text for political variable variants."""
    for var, question in POLITICAL_QUESTIONS.items():
        df.loc[df.variable == var, "question"] = question
    return df
