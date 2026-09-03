import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DEMO_VARS = ["age_c", "cohort_c", "sex", "race_c", "income_c", "degree", "relig_c"]
MISSING_FRAC = 0.10  # remove top 10% by predicted missing probability


def _fit_missing_probs(wide, var, X):
    """Fit logistic regression P(missing | X) for one variable.

    Returns predicted probabilities for observed entries only.
    If fitting fails or there is no variation, returns uniform 0.5.
    """
    is_missing = wide[var].isna()

    if is_missing.all() or (~is_missing).all():
        obs_mask = ~is_missing
        return obs_mask, np.full(obs_mask.sum(), 0.5)

    y = is_missing.astype(int).values

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression(max_iter=300, solver="lbfgs")
            lr.fit(X, y)
            probs = lr.predict_proba(X)[:, 1]
    except Exception:
        probs = np.full(len(y), 0.5)

    obs_mask = ~is_missing
    return obs_mask, probs[obs_mask.values]


def _select_top_missing(yearid_ids, question_id, probs, frac=MISSING_FRAC):
    """Deterministically select top `frac` entries by missing probability.

    Returns DataFrame with (yearid_id, question_id, validation).
    validation=1 for top 10%, validation=0 for the rest.
    """
    n = len(probs)
    n_val = max(1, int(n * frac))

    # Top n_val indices by descending probability
    top_idx = np.argpartition(-probs, n_val)[:n_val]

    validation = np.zeros(n, dtype=int)
    validation[top_idx] = 1

    return pd.DataFrame({
        "yearid_id": yearid_ids,
        "question_id": question_id,
        "validation": validation,
    })


def compute_mar_mask():
    """MAR: missingness modeled from other survey responses, per year.

    Uses variables with <10% missing rate as features (per Appendix B).
    """
    df_analysis = pd.read_pickle(DATA_DIR / "df_analysis.pkl")
    var_qid = (
        df_analysis[["variable", "question_id"]]
        .drop_duplicates("variable")
        .set_index("variable")["question_id"]
    )

    chunks = []
    for year in tqdm(sorted(df_analysis["year"].unique()), desc="MAR"):
        wide = pd.read_parquet(DATA_DIR / f"df_wide_{year}.parquet")
        variables = wide.columns.tolist()

        if len(variables) < 2:
            continue

        # Features: variables with <10% missing rate (>90% observed)
        obs_rate = wide.notna().mean()
        feat_candidates = obs_rate[obs_rate > 0.9].index.tolist()
        wide_feat = wide[feat_candidates].fillna(0.5) if feat_candidates else wide.fillna(0.5)

        for var in variables:
            if var not in var_qid.index:
                continue
            qid = var_qid[var]

            feats = [v for v in feat_candidates if v != var]
            if not feats:
                # Fallback: use all other variables
                feats = [v for v in variables if v != var]

            if all(f in wide_feat.columns for f in feats):
                X = wide_feat[feats].values
            else:
                X = wide[feats].fillna(0.5).values

            obs_mask, probs = _fit_missing_probs(wide, var, X)

            chunks.append(_select_top_missing(
                wide.index[obs_mask].values, qid, probs,
            ))

    return pd.concat(chunks, ignore_index=True)


def compute_mnar_mask():
    """MNAR: missingness modeled from demographic variables, per year."""
    df_analysis = pd.read_pickle(DATA_DIR / "df_analysis.pkl")
    var_qid = (
        df_analysis[["variable", "question_id"]]
        .drop_duplicates("variable")
        .set_index("variable")["question_id"]
    )

    # Load demographics per respondent from gss_qna_all
    demo_raw_cols = [f"demo_{v}" for v in DEMO_VARS]
    gss = pd.read_parquet(
        DATA_DIR / "gss_qna_all.parquet",
        columns=["yearid_year"] + demo_raw_cols,
    )
    gss = gss.rename(columns={f"demo_{v}": v for v in DEMO_VARS})
    resp_demo = gss.groupby("yearid_year")[DEMO_VARS].first()

    # Map yearid_year -> yearid_id
    yid_map = (
        df_analysis[["yearid_year", "yearid_id"]]
        .drop_duplicates("yearid_year")
        .set_index("yearid_year")["yearid_id"]
    )
    resp_demo = resp_demo.join(yid_map, how="inner").set_index("yearid_id")

    chunks = []
    for year in tqdm(sorted(df_analysis["year"].unique()), desc="MNAR"):
        wide = pd.read_parquet(DATA_DIR / f"df_wide_{year}.parquet")
        variables = wide.columns.tolist()

        # Get demographics for respondents in this year
        common_ids = wide.index.intersection(resp_demo.index)
        if len(common_ids) < 10:
            continue

        demo_sub = resp_demo.loc[common_ids].copy()
        demo_sub = pd.get_dummies(demo_sub, drop_first=True).astype(float)
        demo_sub = demo_sub.fillna(0)
        wide_sub = wide.loc[common_ids]
        X = demo_sub.values

        for var in variables:
            if var not in var_qid.index:
                continue
            qid = var_qid[var]

            obs_mask, probs = _fit_missing_probs(wide_sub, var, X)

            chunks.append(_select_top_missing(
                wide_sub.index[obs_mask].values, qid, probs,
            ))

    return pd.concat(chunks, ignore_index=True)


def main():
    print("Computing MAR missing mask...")
    mar = compute_mar_mask()
    n_val = mar["validation"].sum()
    n_total = len(mar)
    out_mar = DATA_DIR / "missing_mask_mar.parquet"
    mar.to_parquet(out_mar, index=False)
    print(f"  {n_val:,} / {n_total:,} = {100*n_val/n_total:.1f}% validation")
    print(f"  Saved -> {out_mar.name}")

    print("\nComputing MNAR missing mask...")
    mnar = compute_mnar_mask()
    n_val = mnar["validation"].sum()
    n_total = len(mnar)
    out_mnar = DATA_DIR / "missing_mask_mnar.parquet"
    mnar.to_parquet(out_mnar, index=False)
    print(f"  {n_val:,} / {n_total:,} = {100*n_val/n_total:.1f}% validation")
    print(f"  Saved -> {out_mnar.name}")


if __name__ == "__main__":
    main()
