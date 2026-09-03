from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.weightstats import DescrStatsW

np.random.seed(1234)
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_gen"
OUT.mkdir(parents=True, exist_ok=True)

DEMO_GROUPS = {
    "Age Group":       "age_c",
    "Gender Group":    "sex",
    "Race Group":      "race_c",
    "Political Group": "polviews",
    "Education Group": "degree",
}


def wmean(g, col):
    v = g[col].values; w = g["wtssall"].values
    mask = ~(np.isnan(v) | np.isnan(w))
    if mask.sum() == 0: return np.nan
    return DescrStatsW(v[mask], weights=w[mask]).mean


def main():
    print("Loading observation + demographics ...")
    df = pd.read_pickle(DATA / "df_analysis_slim.pkl").drop_duplicates(
        ["yearid_id", "question_id"])
    df_full = pd.read_pickle(DATA / "df_analysis.pkl")
    demo_cols = list(DEMO_GROUPS.values())
    demo = df_full[["yearid_id", "yearid"] + demo_cols].drop_duplicates("yearid_id")
    gss = pd.read_parquet(DATA / "gss_df_merged.parquet",
                          columns=["year", "id", "wtssall"])
    df["id"] = df["yearid"] % 10000
    df = df.merge(gss, on=["year", "id"], how="left")

    df_obs = df[["yearid_id", "question_id", "variable", "year", "yearid",
                 "binarized", "wtssall"]].merge(demo, on=["yearid_id", "yearid"],
                                                 how="left")

    var_year = df.groupby("variable")["year"].nunique()
    include = set(var_year[var_year >= 2].index)
    print(f"Multi-year vars: {len(include)}")

    print("Loading partial wide (glm cols only) ...")
    import pyarrow.parquet as pq
    schema = pq.read_schema(PRED / "maicomputer_alpaca-native_partial_10_128_50__resample1_wide.parquet")
    all_cols = [f.name for f in schema]
    glm_cols = [c for c in all_cols if c.endswith("_rescale_logit_glm")
                and c[: -len("_rescale_logit_glm")] in include]
    print(f"  will load {len(glm_cols)} rescale_logit_glm cols")
    wide = pd.read_parquet(
        PRED / "maicomputer_alpaca-native_partial_10_128_50__resample1_wide.parquet",
        columns=["yearid_id"] + glm_cols,
    )

    # Index obs by variable once (used by A10 inner loop).
    obs_by_var = df_obs.groupby("variable")

    # -----------------------------------------------------------------
    # A10: between-group std per variable (weighted mean per group).
    # -----------------------------------------------------------------
    between_rows, between_summary = [], []
    for label, col in DEMO_GROUPS.items():
        print(f"\n--- A10 {label} ({col}) ---")
        bg_pred, bg_obs = [], []
        for i, cn in enumerate(glm_cols):
            var = cn[: -len("_rescale_logit_glm")]
            try:
                obs_sub = obs_by_var.get_group(var)
            except KeyError:
                continue
            obs_sub = obs_sub[["yearid_id", "binarized", "wtssall", col]]
            pred_sub = wide[["yearid_id", cn]].rename(columns={cn: "predicted"})
            m = obs_sub.merge(pred_sub, on="yearid_id", how="inner")
            m = m.dropna(subset=["binarized", "predicted", "wtssall", col])
            if len(m) == 0: continue

            means = m.groupby(col, observed=True).apply(
                lambda g: pd.Series({
                    "pred": wmean(g, "predicted"),
                    "obs":  wmean(g, "binarized"),
                }), include_groups=False
            ).dropna()
            if len(means) >= 2:
                ps = means["pred"].std()
                os_ = means["obs"].std()
                between_rows.append({"demo_group": label, "variable": var,
                                     "pred_std": ps, "obs_std": os_})
                bg_pred.append(ps); bg_obs.append(os_)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(glm_cols)} vars")

        if bg_pred:
            r, _ = spearmanr(bg_pred, bg_obs)
            mae = float(np.mean(np.abs(np.array(bg_pred) - np.array(bg_obs))))
            between_summary.append({"demo_group": label, "r": float(r),
                                     "mae": mae, "n": len(bg_pred)})
            print(f"  between: r={r:.3f} MAE={mae:.4f} n={len(bg_pred)}")

    # -----------------------------------------------------------------
    # A11: within-group std. Vectorized across all variables via
    # long-format melt -> merge -> groupby.
    # -----------------------------------------------------------------
    print("\nBuilding A11 long frame (pred + binarized + demographics) ...")
    pred_long = wide.melt(id_vars="yearid_id",
                          var_name="variable", value_name="predicted")
    pred_long["variable"] = pred_long["variable"].str.replace(
        "_rescale_logit_glm", "", regex=False)
    val_all = pred_long.merge(df_obs, on=["yearid_id", "variable"], how="inner")
    val_all = val_all.dropna(subset=["binarized", "predicted"])
    val_all = val_all[val_all["variable"].isin(include)]
    val_all["predicted_bin"] = (val_all["predicted"] >= 0.5).astype(int)
    print(f"  val_all rows: {len(val_all):,}, vars: {val_all['variable'].nunique()}")

    within_rows_all, within_summary = [], []

    for label, col in DEMO_GROUPS.items():
        print(f"\n--- A11 {label} ({col}) ---")
        sub = val_all.dropna(subset=[col])
        wg = (sub.groupby(["variable", col], observed=True)
                 .agg(pred_std_cont=("predicted", "std"),
                      pred_std_bin =("predicted_bin", "std"),
                      obs_std      =("binarized", "std"),
                      n            =("binarized", "size"))
                 .dropna()
                 .reset_index())
        wg = wg[wg["n"] >= 5]
        if len(wg) == 0:
            continue
        r_cont, _ = spearmanr(wg["pred_std_cont"], wg["obs_std"])
        r_bin,  _ = spearmanr(wg["pred_std_bin"],  wg["obs_std"])
        if r_cont >= r_bin:
            wg["pred_std"] = wg["pred_std_cont"]; mode, r = "continuous", r_cont
        else:
            wg["pred_std"] = wg["pred_std_bin"];  mode, r = "binary",     r_bin
        mae = float((wg["pred_std"] - wg["obs_std"]).abs().mean())

        out = wg.rename(columns={col: "group_id"})[
            ["variable", "group_id", "pred_std", "obs_std"]].copy()
        out["group_id"] = out["group_id"].astype(str)
        out["demo_group"] = label
        out["pred_mode"] = mode
        within_rows_all.append(out[["demo_group", "variable", "group_id",
                                    "pred_std", "obs_std", "pred_mode"]])
        within_summary.append({"demo_group": label, "r": float(r),
                               "mae": mae, "n": int(len(wg)),
                               "pred_mode": mode})
        print(f"  within: r={r:.3f} MAE={mae:.4f} n={len(wg)} ({mode}; "
              f"cont={r_cont:.3f}, bin={r_bin:.3f})")

    # A11 "All" (population-wide, no demographic grouping).
    print("\n--- A11 All (no demo grouping) ---")
    overall = (val_all.groupby("variable", observed=True)
                      .agg(pred_std_cont=("predicted", "std"),
                           pred_std_bin =("predicted_bin", "std"),
                           obs_std      =("binarized", "std"),
                           n            =("binarized", "size"))
                      .dropna().reset_index())
    overall = overall[overall["n"] >= 5]
    r_cont, _ = spearmanr(overall["pred_std_cont"], overall["obs_std"])
    r_bin,  _ = spearmanr(overall["pred_std_bin"],  overall["obs_std"])
    if r_cont >= r_bin:
        overall["pred_std"] = overall["pred_std_cont"]; mode, r = "continuous", r_cont
    else:
        overall["pred_std"] = overall["pred_std_bin"];  mode, r = "binary",     r_bin
    mae = float((overall["pred_std"] - overall["obs_std"]).abs().mean())

    all_out = overall[["variable", "pred_std", "obs_std"]].copy()
    all_out["group_id"] = "all"
    all_out["demo_group"] = "All"
    all_out["pred_mode"] = mode
    within_rows_all.append(all_out[["demo_group", "variable", "group_id",
                                    "pred_std", "obs_std", "pred_mode"]])
    within_summary.append({"demo_group": "All", "r": float(r),
                           "mae": mae, "n": int(len(overall)),
                           "pred_mode": mode})
    print(f"  all: r={r:.3f} MAE={mae:.4f} n={len(overall)} ({mode}; "
          f"cont={r_cont:.3f}, bin={r_bin:.3f})")

    # -------- Save --------
    pd.DataFrame(between_rows).to_parquet(
        OUT / "fig_groupvar_between.parquet", index=False)
    pd.DataFrame(between_summary).to_csv(
        OUT / "fig_groupvar_between_summary.csv", index=False)
    pd.concat(within_rows_all, ignore_index=True).to_parquet(
        OUT / "fig_groupvar_within.parquet", index=False)
    pd.DataFrame(within_summary).to_csv(
        OUT / "fig_groupvar_within_summary.csv", index=False)
    print(f"\nSaved fig_groupvar_{{between,within}}[_summary].* in {OUT}")


if __name__ == "__main__":
    main()
