import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf

np.random.seed(1234)
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_all"
OUT.mkdir(parents=True, exist_ok=True)

MIN_OBS, MIN_POS, MIN_NEG = 10, 2, 2
TASKS = [("impute", "Missing Data Imputation"),
         ("partial", "Retrodiction")]
MODELS = {"alpaca": "maicomputer_alpaca-native", "mf": "mf"}
# 8 demographic groups matching the SM Figure A11 layout:
# Sex, Age, Period, Race, Region, Education, Income, Party ID.
DEMO_GROUPS = ["sex", "age_c", "period", "race_c",
               "region4", "degree", "income_c", "partyid_c"]
SUFFIXES = ("_obs_bin", "_rescale", "_rescale_logit", "_rescale_glm", "_rescale_logit_glm")
RAW_GSS = DATA / "gss7221_r2.dta"

# Census region (1..9) -> 4-way aggregation (East/Midwest/South/West).
# 1 New England, 2 Mid Atlantic -> East
# 3 E.N. Central, 4 W.N. Central -> Midwest
# 5 South Atlantic, 6 E.S. Central, 7 W.S. Central -> South
# 8 Mountain, 9 Pacific -> West
REGION4_MAP = {1: "East", 2: "East",
               3: "Midwest", 4: "Midwest",
               5: "South", 6: "South", 7: "South",
               8: "West", 9: "West"}

# GSS partyid codes 0..7 -> SM labels (Independent is reference level).
PARTYID_MAP = {
    0: "Strong Democrat",
    1: "Weak Democrat",
    2: "Lean Democrat",
    3: "Independent",
    4: "Lean Republican",
    5: "Weak Republican",
    6: "Strong Republican",
    7: "Something else",
}


def _bare_pred_cols(parquet_path):
    cols = [f.name for f in pq.read_schema(parquet_path)]
    return [c for c in cols if c != "yearid_id"
            and not any(c.endswith(s) for s in SUFFIXES)]


def _groupwise_auc(y, s, group_ids):
    """Vectorised Mann-Whitney AUC per-group.

    AUC = (sum_of_ranks_of_positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg).
    Returns a dict {group_id: (auc, n)} for groups meeting MIN_* thresholds.
    """
    # sort by (group, score) for group-local ranking
    order = np.lexsort((s, group_ids))
    g_ord = group_ids[order]
    y_ord = y[order]

    # group boundaries
    bnd = np.concatenate(([0], np.flatnonzero(g_ord[1:] != g_ord[:-1]) + 1,
                          [len(g_ord)]))

    out = {}
    for i in range(len(bnd) - 1):
        lo, hi = bnd[i], bnd[i + 1]
        yg = y_ord[lo:hi]
        n = hi - lo
        n_pos = int(yg.sum())
        n_neg = n - n_pos
        if n < MIN_OBS or n_pos < MIN_POS or n_neg < MIN_NEG:
            continue
        # ranks within the group (1..n)
        # lo..hi is already sorted by score
        idx = np.arange(lo + 1, hi + 1)
        ranks_pos_sum = idx[yg == 1].sum() - lo * n_pos
        auc = (ranks_pos_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        out[int(g_ord[lo])] = (float(auc), int(n))
    return out


def individual_aucs(wide_path, df_obs):
    """Compute per-respondent AUC pooled across that respondent's (var) obs."""
    t0 = time.time()
    bare = _bare_pred_cols(wide_path)
    print(f"    loading {wide_path.name}  ({len(bare)} pred cols)")
    wide = pd.read_parquet(wide_path, columns=["yearid_id"] + bare)
    # wide has ~68k rows; yearid_id is unique
    wide = wide.set_index("yearid_id")
    wide_arr = wide.to_numpy(copy=False)         # (n_yid, n_var)
    var_to_col = {v: j for j, v in enumerate(bare)}
    yid_to_row = pd.Series(np.arange(len(wide)), index=wide.index)
    print(f"    wide ready in {time.time()-t0:.1f}s")

    # Filter obs to variables actually in this wide parquet
    mask_var = df_obs["variable"].isin(var_to_col)
    sub = df_obs.loc[mask_var, ["yearid_id", "variable", "binarized"]]
    # Map to row/col indices
    row_idx = yid_to_row.reindex(sub["yearid_id"].values).to_numpy()
    col_idx = sub["variable"].map(var_to_col).to_numpy()
    keep = pd.notna(row_idx) & pd.notna(col_idx)
    row_idx = row_idx[keep].astype(np.int64)
    col_idx = col_idx[keep].astype(np.int64)
    yids_kept = sub["yearid_id"].to_numpy()[keep]
    bins_kept = sub["binarized"].to_numpy()[keep].astype(np.int8)

    pred = wide_arr[row_idx, col_idx]            # vectorised lookup
    finite = np.isfinite(pred) & ~pd.isna(bins_kept)
    y = bins_kept[finite]
    s = pred[finite]
    g = yids_kept[finite].astype(np.int64)
    print(f"    {len(y):,} obs-prediction pairs in {time.time()-t0:.1f}s")

    aucs = _groupwise_auc(y, s, g)
    print(f"    {len(aucs):,} respondents with valid AUC in {time.time()-t0:.1f}s")

    rows = [{"yearid_id": yid, "auc": a, "n": n} for yid, (a, n) in aucs.items()]
    return pd.DataFrame(rows)


REFERENCE_LEVELS = {
    "sex":       "male",
    "age_c":     "18-24",
    "period":    "1970s",
    "race_c":    "White",
    "region4":   "East",
    "degree":    "less than high school",
    "income_c":  "-15000",
    "partyid_c": "Independent",
}


def fit_regression(df, task_label, model_name):
    d = df.dropna(subset=["auc"] + DEMO_GROUPS).copy()
    print(f"    fit_regression rows: {len(d):,}")
    if len(d) < 50:
        return []
    terms = " + ".join(
        f"C({g}, Treatment(reference='{REFERENCE_LEVELS[g]}'))"
        if g in REFERENCE_LEVELS else f"C({g})"
        for g in DEMO_GROUPS
    )
    try:
        mod = smf.ols(f"auc ~ {terms}", data=d).fit()
    except Exception as e:
        print(f"    regression failed ({model_name}/{task_label}): {e}")
        return []
    out = []
    for name in mod.params.index:
        if name == "Intercept":
            continue
        # Match both plain `C(col)[T.level]` and
        # `C(col, Treatment(reference='...'))[T.level]`. The Treatment
        # clause contains its own `)`, so accept anything up to `)[T.`.
        m = re.match(r"^C\((\w+)(?:,.*)?\)\[T\.(.+)\]$", name)
        if not m:
            continue
        g, lvl = m.group(1), m.group(2)
        out.append({
            "task": task_label, "demo_group": g, "level": lvl, "model": model_name,
            "coef": float(mod.params[name]), "se": float(mod.bse[name]),
            "p": float(mod.pvalues[name]),
        })
    return out


def _build_demographics():
    """One row per yearid_id with the 8 SM demo columns.

    Pulls region and partyid from the raw GSS Stata file (not present in
    df_analysis.pkl) and collapses them to SM-style categories. Derives a
    decade period from year.
    """
    df_full = pd.read_pickle(DATA / "df_analysis.pkl")
    base = df_full[["yearid_id", "yearid", "year",
                    "age_c", "sex", "race_c", "income_c", "degree"]
                   ].drop_duplicates("yearid_id").copy()

    # Period: decade bands. Upper bins are right-inclusive by default.
    base["period"] = pd.cut(base["year"],
                            bins=[1971, 1980, 1990, 2000, 2010, 2021],
                            labels=["1970s", "1980s", "1990s", "2000s", "2010s"])
    base["period"] = base["period"].astype("object")

    # Raw GSS: region + partyid keyed by yearid (= year || id.zfill(4)).
    raw = pd.read_stata(RAW_GSS,
                        columns=["year", "id", "region", "partyid"],
                        convert_categoricals=False)
    raw["yearid"] = (raw["year"].astype(int).astype(str)
                     + raw["id"].fillna(0).astype(int).astype(str).str.zfill(4))
    raw["yearid"] = pd.to_numeric(raw["yearid"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["yearid"])
    raw["region4"] = raw["region"].map(REGION4_MAP)
    raw["partyid_c"] = raw["partyid"].map(PARTYID_MAP)
    raw = raw[["yearid", "region4", "partyid_c"]].drop_duplicates("yearid")

    base["yearid"] = pd.to_numeric(base["yearid"], errors="coerce").astype("Int64")
    demo = base.merge(raw, on="yearid", how="left")
    return demo[["yearid_id"] + DEMO_GROUPS]


def main():
    t0 = time.time()
    print("Loading obs + demographics ...")
    df = pd.read_pickle(DATA / "df_analysis_slim.pkl").drop_duplicates(
        ["yearid_id", "question_id"])
    demo = _build_demographics()
    df_obs = df[["yearid_id", "variable", "binarized"]].copy()
    df_obs = df_obs.dropna(subset=["binarized"])
    print(f"  obs rows: {len(df_obs):,}  ({time.time()-t0:.1f}s)")
    print(f"  demo rows: {len(demo):,}  "
          + ", ".join(f"{g}: n_null={demo[g].isna().sum()}"
                      for g in DEMO_GROUPS))

    all_rows = []
    for split, task_label in TASKS:
        print(f"\n=== {task_label} ({split}) ===")
        per_model = {}
        for mk, mname in MODELS.items():
            wide_path = PRED / f"{mname}_{split}_10_128_50__resample1_wide.parquet"
            if not wide_path.exists():
                print(f"  [MISS] {wide_path.name}")
                continue
            print(f"  -- {mk} --")
            ia = individual_aucs(wide_path, df_obs)
            per_model[mk] = ia.merge(demo, on="yearid_id", how="left")

        if "alpaca" not in per_model or "mf" not in per_model:
            continue

        rows_a = fit_regression(per_model["alpaca"], task_label, "alpaca")
        rows_m = fit_regression(per_model["mf"], task_label, "mf")
        df_a, df_m = pd.DataFrame(rows_a), pd.DataFrame(rows_m)
        if df_a.empty or df_m.empty:
            continue
        merged = df_a.merge(df_m, on=["task", "demo_group", "level"],
                            how="outer", suffixes=("_alpaca", "_mf"))
        merged["alpaca_sig"] = merged["p_alpaca"] < 0.05
        merged["mf_sig"] = merged["p_mf"] < 0.05
        out = merged.rename(columns={
            "coef_alpaca": "alpaca_coef", "se_alpaca": "alpaca_se",
            "coef_mf": "mf_coef", "se_mf": "mf_se",
        })[
            ["task", "demo_group", "level",
             "alpaca_coef", "alpaca_se", "alpaca_sig",
             "mf_coef", "mf_se", "mf_sig"]
        ]
        out["n_respondents"] = int(min(len(per_model["alpaca"]),
                                       len(per_model["mf"])))
        all_rows.append(out)

    if not all_rows:
        print("No results produced.")
        return
    final = pd.concat(all_rows, ignore_index=True)
    out_path = OUT / "figA16_individual_auc_regression.parquet"
    final.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(final)} rows)  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
