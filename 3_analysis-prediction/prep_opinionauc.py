import re
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import pyarrow.parquet as pq
import statsmodels.formula.api as smf

np.random.seed(1234)
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_gen"
OUT.mkdir(parents=True, exist_ok=True)

MIN_OBS, MIN_POS, MIN_NEG = 10, 2, 2
TASKS = [("impute", "Missing Data Imputation"),
         ("partial", "Retrodiction"),
         ("total", "Unasked Opinion Prediction")]
MODEL = "maicomputer_alpaca-native"
SUFFIXES = ("_obs_bin", "_rescale", "_rescale_logit", "_rescale_glm",
            "_rescale_logit_glm")
RAW_GSS = DATA / "gss7221_r2.dta"

# GSS polviews 7-point numeric map.
POLVIEWS_MAP = {
    "extremely liberal": 1,
    "liberal": 2,
    "slightly liberal": 3,
    "moderate, middle of the road": 4,
    "moderate": 4,
    "slightly conservative": 5,
    "conservative": 6,
    "extremely conservative": 7,
}

# Historical GSS response rates (NORC/GSS publications). Dictionary keyed by
# year. If a year is missing we impute with the series mean.
GSS_RESPONSE_RATE = {
    1972: 0.759, 1973: 0.772, 1974: 0.752, 1975: 0.758, 1976: 0.751,
    1977: 0.752, 1978: 0.732, 1980: 0.755, 1982: 0.776, 1983: 0.779,
    1984: 0.791, 1985: 0.788, 1986: 0.747, 1987: 0.756, 1988: 0.773,
    1989: 0.773, 1990: 0.739, 1991: 0.775, 1993: 0.820, 1994: 0.779,
    1996: 0.761, 1998: 0.757, 2000: 0.702, 2002: 0.702, 2004: 0.703,
    2006: 0.712, 2008: 0.701, 2010: 0.702, 2012: 0.714, 2014: 0.612,
    2016: 0.614, 2018: 0.596, 2021: 0.172,
}

FEATURES_ORDER = [
    ("t",                   "Survey years"),
    ("year_2021",           "Year = 2021"),
    ("response_rate",       "Yearly Response Rate"),
    ("condition_ballot",    "Under Split Ballot Design"),
    ("n_year",              "N of collected survey periods"),
    ("n_response",          "N individual response for opinion"),
    ("two_category",        "binary response (vs three or more)"),
    ("p_dk",                "P(Do not know response)"),
    ("p_skip",              "P(Skip response)"),
    ("p_refuse",            "P(Refuse response)"),
    ("p_inapplicable",      "P(Inapplicable response)"),
    ("abs_corr_with_ideo",  "Absolute Corr. with political ideology"),
    ("var_response",        "Variance of response"),
    ("cos_sim",             "Average Cosine Similarity from Other Opinions"),
]
CONTINUOUS = {"t", "response_rate", "n_response", "p_dk", "p_skip",
              "p_refuse", "p_inapplicable", "abs_corr_with_ideo",
              "var_response", "cos_sim"}
# n_year stays on the natural scale.


def _bare_pred_cols(parquet_path):
    cols = [f.name for f in pq.read_schema(parquet_path)]
    return [c for c in cols if c != "yearid_id"
            and not any(c.endswith(s) for s in SUFFIXES)]


def _groupwise_auc(y, s, group_ids):
    """Vectorised Mann-Whitney AUC per group.

    Returns dict {gid: (auc, n)}.
    """
    order = np.lexsort((s, group_ids))
    g_ord = group_ids[order]
    y_ord = y[order]
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
        idx = np.arange(lo + 1, hi + 1)
        ranks_pos_sum = idx[yg == 1].sum() - lo * n_pos
        auc = (ranks_pos_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        out[int(g_ord[lo])] = (float(auc), int(n))
    return out


def pooled_var_year_auc(wide_path, df_obs):
    """Compute per-(variable, year) pooled AUC.

    Groups observations by a compact (var_id, year) key so a single call to
    `_groupwise_auc` handles every group across all variables at once.
    """
    t0 = time.time()
    bare = _bare_pred_cols(wide_path)
    print(f"    loading {wide_path.name}  ({len(bare)} pred cols)")
    wide = pd.read_parquet(wide_path, columns=["yearid_id"] + bare)
    wide = wide.set_index("yearid_id")
    wide_arr = wide.to_numpy(copy=False)
    var_to_col = {v: j for j, v in enumerate(bare)}
    yid_to_row = pd.Series(np.arange(len(wide)), index=wide.index)
    print(f"    wide ready in {time.time() - t0:.1f}s")

    mask_var = df_obs["variable"].isin(var_to_col)
    sub = df_obs.loc[mask_var, ["yearid_id", "variable", "year", "binarized"]]
    row_idx = yid_to_row.reindex(sub["yearid_id"].values).to_numpy()
    col_idx = sub["variable"].map(var_to_col).to_numpy()
    keep = pd.notna(row_idx) & pd.notna(col_idx)
    row_idx = row_idx[keep].astype(np.int64)
    col_idx = col_idx[keep].astype(np.int64)
    variables_kept = sub["variable"].to_numpy()[keep]
    years_kept = sub["year"].to_numpy()[keep]
    bins_kept = sub["binarized"].to_numpy()[keep].astype(np.int8)

    pred = wide_arr[row_idx, col_idx]
    finite = np.isfinite(pred) & ~pd.isna(bins_kept)
    variables_kept = variables_kept[finite]
    years_kept = years_kept[finite]
    y = bins_kept[finite]
    s = pred[finite]

    # compact group id: (variable, year) -> int
    g_df = pd.DataFrame({"variable": variables_kept,
                         "year": years_kept.astype(np.int32)})
    g_df["gid"] = pd.factorize(
        list(zip(g_df["variable"].to_numpy(), g_df["year"].to_numpy())),
        sort=False)[0]
    g = g_df["gid"].to_numpy().astype(np.int64)
    lookup = g_df.drop_duplicates("gid").set_index("gid")[["variable", "year"]]
    print(f"    {len(y):,} obs-prediction pairs, "
          f"{g_df['gid'].nunique():,} (var, year) cells in "
          f"{time.time() - t0:.1f}s")

    aucs = _groupwise_auc(y, s, g)
    rows = []
    for gid, (a, n) in aucs.items():
        v, yr = lookup.loc[gid]
        rows.append({"variable": v, "year": int(yr), "auc": float(a), "n": int(n)})
    out = pd.DataFrame(rows)
    print(f"    {len(out):,} (var, year) groups with valid AUC in "
          f"{time.time() - t0:.1f}s")
    return out


def count_categories(code_labels):
    if pd.isna(code_labels) or not isinstance(code_labels, str):
        return None
    parts = re.split(r"[\n,;|]", code_labels)
    parts = [p.strip() for p in parts if p.strip()
             and not p.strip().startswith(".")]
    return len(parts) if parts else None


def build_static_features(all_vars):
    """Features that depend only on variable (not year):
      n_year (per-variable count of survey years with obs),
      two_category (1 if n_category == 2),
      cos_sim (mean cosine similarity with all other variables).
    """
    # n_year from obs count
    df_obs = pd.read_pickle(DATA / "df_analysis_slim.pkl").drop_duplicates(
        ["yearid_id", "question_id"])
    n_year = (df_obs.dropna(subset=["binarized"])
                    .groupby("variable")["year"].nunique()
                    .rename("n_year").reset_index())

    # n_category
    varmeta = pd.read_parquet(
        DATA / "gss_train_vars_corrected_binarized_again.parquet")
    varmeta["n_cat"] = varmeta["code_labels"].apply(count_categories)
    nc = varmeta[["variable_name", "n_cat"]].rename(
        columns={"variable_name": "variable", "n_cat": "n_category"})
    nc["two_category"] = (nc["n_category"] == 2).astype(int)

    # cosine similarity via question embeddings (Alpaca-7b mean-pooled)
    emb_path = DATA / "weights" / f"{MODEL}.pkl"
    emb = pickle.load(open(emb_path, "rb"))
    emb = np.vstack(emb) if isinstance(emb, list) else np.asarray(emb)
    # Map question_id -> variable (one variable can have several question_ids,
    # but most have one). Average embeddings per variable.
    df_full = pd.read_pickle(DATA / "df_analysis_slim.pkl")
    qv = df_full[["question_id", "variable"]].drop_duplicates("question_id")
    qv = qv.sort_values("question_id").reset_index(drop=True)
    # emb rows aligned by question_id
    max_qid = int(qv["question_id"].max())
    if emb.shape[0] <= max_qid:
        raise RuntimeError(f"Embedding rows {emb.shape[0]} <= max qid {max_qid}")
    emb_for_q = emb[qv["question_id"].to_numpy()]
    var_vec = (pd.DataFrame(emb_for_q)
                 .assign(variable=qv["variable"].to_numpy())
                 .groupby("variable").mean())
    V = var_vec.to_numpy(dtype=np.float32)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    sim = Vn @ Vn.T                       # (nv, nv)
    # exclude self similarity (diagonal)
    np.fill_diagonal(sim, 0.0)
    mean_sim = sim.sum(axis=1) / (sim.shape[1] - 1)
    cos_df = pd.DataFrame({"variable": var_vec.index.to_numpy(),
                           "cos_sim": mean_sim})

    static = (pd.DataFrame({"variable": sorted(all_vars)})
                .merge(n_year, on="variable", how="left")
                .merge(nc[["variable", "n_category", "two_category"]],
                       on="variable", how="left")
                .merge(cos_df, on="variable", how="left"))
    return static


def _aggregate_stata_columns(year, ballot_has, cols_batch, raw_df):
    """Compute per-(year, variable) missing-code breakdown for a batch of
    Stata columns. Returns a list of dicts (one per (var, year) cell).

    Uses pandas groupby by year, computing per-code counts via boolean-masked
    ndarrays to avoid O(years*cols) Python-level masks.
    """
    year_arr = year.to_numpy()
    ballot_has_arr = ballot_has.to_numpy().astype(np.int32)
    # sort by year once for groupby-like bookkeeping
    order = np.argsort(year_arr, kind="stable")
    y_sorted = year_arr[order]
    ballot_sorted = ballot_has_arr[order]
    # boundaries per year
    bnd = np.concatenate(([0], np.flatnonzero(y_sorted[1:] != y_sorted[:-1]) + 1,
                          [len(y_sorted)]))
    years = [int(y_sorted[b]) for b in bnd[:-1]]

    out_rows = []
    for col in cols_batch:
        arr = raw_df[col].astype(str).to_numpy()[order]
        is_dk    = (arr == ".d")
        is_ref   = (arr == ".n")
        is_inapp = (arr == ".i")
        is_skip  = (arr == ".s")
        is_drop  = (arr == ".x") | (arr == ".y") | (arr == ".m") | \
                   (arr == "nan") | (arr == "None")
        is_valid = ~(is_dk | is_ref | is_inapp | is_skip | is_drop)
        kept = ~is_drop
        for i in range(len(bnd) - 1):
            lo, hi = bnd[i], bnd[i + 1]
            n_valid = int(is_valid[lo:hi].sum())
            if n_valid == 0:
                continue
            n_kept = int(kept[lo:hi].sum())
            out_rows.append({
                "variable": col, "year": years[i],
                "n_total": n_kept, "n_valid": n_valid,
                "n_dk": int(is_dk[lo:hi].sum()),
                "n_refuse": int(is_ref[lo:hi].sum()),
                "n_inapp": int(is_inapp[lo:hi].sum()),
                "n_skip": int(is_skip[lo:hi].sum()),
                "n_ballot": int((ballot_sorted[lo:hi] & kept[lo:hi]).sum()),
            })
    return out_rows


def build_varyear_features():
    """Features that vary by (variable, year):
      condition_ballot, p_dk, p_skip, p_refuse, p_inapplicable,
      var_response, mean_response, abs_corr_with_ideo, n_response.

    Stata raw is read in column batches (full rows but a slice of variables
    per pass) to cap peak memory while keeping convert_missing=True so the
    .d/.n/.i/.s codes survive.
    """
    varmeta = pd.read_parquet(
        DATA / "gss_train_vars_corrected_binarized_again.parquet")
    model_vars = sorted(set(varmeta["variable_name"].dropna().astype(str)))

    # probe the full Stata header once via variable_labels()
    vrdr = pd.io.stata.StataReader(str(RAW_GSS))
    stata_cols = set(vrdr.variable_labels().keys())
    # exclude year/id/ballot from the batched columns (they're handled by
    # the anchor read).
    target_cols = [c for c in model_vars
                   if c in stata_cols and c not in ("year", "id", "ballot")]
    print(f"  Stata cols in model vars: {len(target_cols)}")

    # read the anchor columns (year, id, ballot) once
    t0 = time.time()
    anchor = pd.read_stata(str(RAW_GSS),
                           columns=["year", "id", "ballot"],
                           convert_missing=True,
                           convert_categoricals=False)
    print(f"    anchor read {anchor.shape} in {time.time() - t0:.1f}s")
    year = anchor["year"].astype(int)
    # IMPORTANT: with convert_missing=True, Stata extended missing codes
    # (.i/.n/.d/.s/.x/.y/.m) are NOT detected by .isna(). Treat any non-numeric
    # string repr as missing so that pre-1988 GSS (where ballot is .i for all)
    # correctly scores as "no split ballot".
    _ballot_strs = anchor["ballot"].astype(str).to_numpy()
    _ballot_missing_codes = {"nan", ".d", ".n", ".i", ".s",
                             ".x", ".y", ".m", "None"}
    ballot_has = pd.Series(
        ~np.isin(_ballot_strs, list(_ballot_missing_codes)),
        index=anchor.index,
    )

    # batch through the rest
    BATCH = 400
    all_rows = []
    n_batches = (len(target_cols) + BATCH - 1) // BATCH
    for bi in range(n_batches):
        batch = target_cols[bi * BATCH:(bi + 1) * BATCH]
        t1 = time.time()
        # year included to anchor rows
        raw = pd.read_stata(str(RAW_GSS),
                            columns=["year"] + batch,
                            convert_missing=True,
                            convert_categoricals=False)
        agg_rows = _aggregate_stata_columns(year, ballot_has, batch, raw)
        all_rows.extend(agg_rows)
        print(f"    batch {bi+1}/{n_batches} "
              f"({len(batch)} cols) in {time.time() - t1:.1f}s, "
              f"cum rows={len(all_rows):,}")
        del raw

    agg = pd.DataFrame(all_rows)
    agg["p_dk"] = agg["n_dk"] / agg["n_total"]
    agg["p_refuse"] = agg["n_refuse"] / agg["n_total"]
    agg["p_skip"] = agg["n_skip"] / agg["n_total"]
    agg["p_inapplicable"] = agg["n_inapp"] / agg["n_total"]
    agg["p_ballot"] = agg["n_ballot"] / agg["n_total"]
    agg["condition_ballot"] = (agg["p_ballot"] > 0).astype(int)
    print(f"  aggregated {len(agg):,} (year, variable) missing-code cells")

    # 2) Additionally, compute var_response, mean_response, n_response, and
    #    abs_corr_with_ideo from df_analysis_slim + df_analysis polviews.
    print("  loading df_analysis + polviews ...")
    df_slim = pd.read_pickle(DATA / "df_analysis_slim.pkl").drop_duplicates(
        ["yearid_id", "question_id"])
    df_full = pd.read_pickle(DATA / "df_analysis.pkl")
    polv = df_full[["yearid_id", "polviews"]].drop_duplicates("yearid_id")
    polv["polv_num"] = polv["polviews"].astype(str).str.lower().map(
        POLVIEWS_MAP)
    df_slim = df_slim.merge(polv[["yearid_id", "polv_num"]],
                            on="yearid_id", how="left")

    gb = (df_slim.dropna(subset=["binarized"])
                 .groupby(["variable", "year"]))
    stats = gb["binarized"].agg(["mean", "var", "size"])
    stats.columns = ["mean_response", "var_response", "n_response"]
    stats = stats.reset_index()
    stats["year"] = stats["year"].astype(int)

    # abs_corr_with_ideo
    def _abs_corr(g):
        d = g.dropna(subset=["polv_num", "binarized"])
        if len(d) < 5: return np.nan
        if d["binarized"].std() == 0 or d["polv_num"].std() == 0:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(abs(d["binarized"].corr(d["polv_num"])))

    print("  computing per-(var, year) |corr with polviews| ...")
    t4 = time.time()
    corr = gb.apply(_abs_corr, include_groups=False).rename(
        "abs_corr_with_ideo").reset_index()
    corr["year"] = corr["year"].astype(int)
    print(f"    done in {time.time() - t4:.1f}s")

    varyear = (stats.merge(corr, on=["variable", "year"], how="left")
                    .merge(agg, on=["variable", "year"], how="outer"))
    return varyear


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd is None or sd == 0 or np.isnan(sd):
        return s - mu
    return (s - mu) / sd


def fit_and_extract(df, task_label):
    feats = [f for f, _ in FEATURES_ORDER]
    d = df.dropna(subset=["auc"] + feats).copy()
    print(f"    rows for OLS: {len(d):,}")
    if len(d) < 30:
        print(f"    [WARN] too few rows for task {task_label}; skipping")
        return []
    formula = "auc ~ " + " + ".join(feats)
    try:
        res = smf.ols(formula, data=d).fit(cov_type="HC1")
    except Exception as e:
        print(f"    OLS failed for {task_label}: {e}")
        return []
    rows = []
    for f, lbl in FEATURES_ORDER:
        if f not in res.params.index: continue
        p = float(res.pvalues[f])
        rows.append({
            "task": task_label,
            "feature": f,
            "label": lbl,
            "coef": float(res.params[f]),
            "se": float(res.bse[f]),
            "p": p,
            "sig": p < 0.05,
            "n_groups": int(len(d)),
        })
    return rows


def main():
    t0 = time.time()
    print("=== load observations ===")
    df_obs = (pd.read_pickle(DATA / "df_analysis_slim.pkl")
                .drop_duplicates(["yearid_id", "question_id"])
                [["yearid_id", "variable", "year", "binarized"]]
                .dropna(subset=["binarized"]))
    print(f"  df_obs rows: {len(df_obs):,}")

    print("\n=== build per-(var, year) features ===")
    var_year_feat = build_varyear_features()
    print(f"  var-year features: {len(var_year_feat):,} rows")

    print("\n=== build static (per-variable) features ===")
    all_vars = sorted(set(var_year_feat["variable"]))
    static_feat = build_static_features(all_vars)
    print(f"  static features: {len(static_feat):,} rows")

    # annual response rate
    rr = pd.DataFrame([{"year": y, "response_rate": v}
                       for y, v in GSS_RESPONSE_RATE.items()])

    all_rows = []
    for split, task_label in TASKS:
        print(f"\n=== task={split} ({task_label}) ===")
        wide_path = PRED / f"{MODEL}_{split}_10_128_50__resample1_wide.parquet"
        if not wide_path.exists():
            print(f"  [MISS] {wide_path.name}; skipping task")
            continue
        auc_df = pooled_var_year_auc(wide_path, df_obs)
        if auc_df.empty:
            print(f"  [WARN] no valid AUC cells for task {split}")
            continue
        merged = (auc_df.merge(static_feat, on="variable", how="left")
                        .merge(var_year_feat, on=["variable", "year"],
                               how="left")
                        .merge(rr, on="year", how="left"))
        merged["year_2021"] = (merged["year"] == 2021).astype(int)
        merged["t"] = merged["year"].astype(float)
        # condition_ballot NA -> 0 (interpret missing ballot column as "not split")
        merged["condition_ballot"] = merged["condition_ballot"].fillna(0).astype(int)
        # var_response & abs_corr_with_ideo are multiplied by 100 then scaled
        merged["var_response"] = merged["var_response"] * 100
        merged["abs_corr_with_ideo"] = merged["abs_corr_with_ideo"] * 100
        # z-score continuous features
        for c in CONTINUOUS:
            merged[c] = zscore(merged[c])
        merged.insert(0, "task", task_label)
        feat_cols = [f for f, _ in FEATURES_ORDER]
        keep_cols = ["task", "variable", "year", "auc"] + feat_cols
        all_rows.append(merged[keep_cols].copy())

    if not all_rows:
        print("\n[FATAL] no per-varyear rows produced.")
        return
    out = pd.concat(all_rows, ignore_index=True)
    out_path = OUT / "opinionauc_per_varyear.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path} ({len(out):,} rows) total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
