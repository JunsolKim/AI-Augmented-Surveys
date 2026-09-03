import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = DATA / "fig_table_gen"
GEMMA = DATA / "roper_prep" / "results" / "colab_llm_gemma"
META = DATA / "roper_prep" / "roper_question_meta.parquet"
PROMPT_B_4VARS = DATA / "verify_prompt_b_4vars.csv"

TARGET_VARS = ["marhomo1", "busing", "concong", "cohabit"]


def _is_simple_binary(bm: str) -> bool:
    """Accept binarization mappings whose values are all in {0, 1, None} and
    whose keys are simple option labels. Reject compound keys formed by joining
    distinct response prefixes with '-' or 'and' — those mark filter/skip-logic
    collapses that misalign with the GSS question being matched."""
    try:
        m = json.loads(bm)
    except Exception:
        return False
    if not all(v in (0, 1, None) for v in m.values()):
        return False
    for k in m.keys():
        parts = k.split("-")
        if len(parts) >= 2 and len(k) > 25:
            return False
        if " and " in k.lower() and len(k) > 25:
            return False
    return True


def load_roper_polls() -> pd.DataFrame:
    """One row per (variable, Roper question, year) passing the inclusion
    rule, with `_high` / `_marg` flags for the two regimes."""
    cols = ["gss_variable", "year", "roper_question", "roper_question_id",
            "cos_sim", "roper_yes_pct", "llm_mapping"]
    bo = pd.read_csv(GEMMA / "binarize_output.csv", usecols=cols)
    bof = pd.read_csv(GEMMA / "binarize_output_fill.csv", usecols=cols)
    bin_df = pd.concat([bo, bof], ignore_index=True).drop_duplicates(
        ["gss_variable", "roper_question_id", "year"])

    vo = pd.read_csv(
        GEMMA / "verify_output.csv",
        usecols=["gss_variable", "roper_question", "llm_same", "llm_confidence"],
    )
    vo = (vo.loc[vo["llm_same"] == "Yes",
                 ["gss_variable", "roper_question", "llm_confidence"]]
          .drop_duplicates(["gss_variable", "roper_question"]))

    meta = pd.read_parquet(META)[
        ["roper_question_id", "national_adult", "study_title",
         "survey_by", "conducted_by"]]
    df = bin_df.merge(vo, on=["gss_variable", "roper_question"], how="inner")
    df = df.merge(meta, on="roper_question_id", how="left")
    df["national_adult"] = df["national_adult"].fillna(0).astype(int)
    df["simple_bin"] = df["llm_mapping"].astype(str).apply(_is_simple_binary)
    df["roper_mean"] = pd.to_numeric(df["roper_yes_pct"], errors="coerce") / 100.0
    df["year"] = df["year"].astype(int)

    # Prompt B re-check (only used for the marginal-cos regime)
    if PROMPT_B_4VARS.exists():
        pb = pd.read_csv(PROMPT_B_4VARS, usecols=[
            "gss_variable", "roper_question", "B_answer"])
        pb_yes = (pb.loc[pb["B_answer"] == "Yes",
                         ["gss_variable", "roper_question"]]
                  .drop_duplicates())
        pb_yes["B_yes"] = True
        df = df.merge(pb_yes, on=["gss_variable", "roper_question"], how="left")
        df["B_yes"] = df["B_yes"].fillna(False).astype(bool)
    else:
        df["B_yes"] = False

    base = (df["national_adult"] == 1) & df["simple_bin"] \
           & df["roper_mean"].between(0.0, 1.0)
    high = base & (df["cos_sim"] >= 0.85) & (df["llm_confidence"] >= 0.85)
    marg = base & (df["cos_sim"] >= 0.80) & (df["cos_sim"] < 0.85) \
           & df["B_yes"]

    df = df.assign(_high=high, _marg=marg)
    df = df.loc[df["_high"] | df["_marg"]]

    # Restrict to year <= 2021 (we only model 1972-2021).
    return df.loc[df["year"] <= 2021]


def select_polls(df: pd.DataFrame, gss_years_by_var: dict) -> pd.DataFrame:
    """Apply the per-variable standard/fallback choice and drop GSS years."""
    out = []
    for var in TARGET_VARS:
        gss_years = gss_years_by_var.get(var, set())
        var_df = df[(df["gss_variable"] == var) & ~df["year"].isin(gss_years)]
        std = var_df[var_df["_high"]]
        # FALLBACK: only when the standard filter returns zero polls for the
        # variable do we admit the marginal-cos Prompt-B-verified polls.
        if len(std) > 0:
            sub = std
        else:
            sub = var_df[var_df["_marg"]]
            if len(sub) > 0:
                print(f'  FALLBACK for "{var}": standard filter empty; '
                      f'admitting {len(sub)} marginal polls verified by '
                      f'construct-aware Prompt B')
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    obs_slim = (pd.read_pickle(DATA / "df_analysis_slim.pkl")
                [["variable", "year"]].drop_duplicates())
    gss_years_by_var = obs_slim.groupby("variable")["year"].apply(set).to_dict()

    polls = select_polls(load_roper_polls(), gss_years_by_var)

    # Per-(variable, year) means for Figure 5.
    roper = pd.concat([
        polls[polls["gss_variable"] == var]
        .groupby(["gss_variable", "year"])
        .agg(roper_mean=("roper_mean", "mean"),
             n_polls=("roper_mean", "size"))
        .reset_index()
        for var in TARGET_VARS
    ], ignore_index=True).rename(columns={"gss_variable": "variable"})
    roper_out = OUT / "counterfactual_roper.parquet"
    roper.to_parquet(roper_out, index=False)
    print(f"Saved: {roper_out} ({len(roper)} Roper non-GSS points)")
    for var in TARGET_VARS:
        s = roper[roper["variable"] == var]
        print(f"  {var}: {len(s)} Roper points, years {sorted(s['year'].tolist())}")

    # Per-poll listing for Table A12: one row per (variable, year, study) —
    # many gemma rows have the same study but different roper_question
    # phrasings; collapse to study level.
    surveys = (polls
               .sort_values(["gss_variable", "year", "study_title", "cos_sim"],
                            ascending=[True, True, True, False])
               .drop_duplicates(["gss_variable", "year", "study_title"])
               [["gss_variable", "year", "study_title", "survey_by",
                 "conducted_by", "cos_sim", "llm_confidence", "roper_mean"]]
               .rename(columns={"study_title": "studyTitle",
                                "survey_by": "surveyBy",
                                "conducted_by": "conductedBy"})
               .sort_values(["gss_variable", "year", "studyTitle"])
               .reset_index(drop=True))
    surveys_out = OUT / "counterfactual_roper_surveys.parquet"
    surveys.to_parquet(surveys_out, index=False)
    print(f"Saved: {surveys_out} ({len(surveys)} polls)")


if __name__ == "__main__":
    main()
