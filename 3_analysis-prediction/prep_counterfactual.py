from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW


np.random.seed(1234)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_gen"

TARGET_VARS = ["marhomo1", "busing", "concong", "cohabit"]
MODELS = [("alpaca", "maicomputer_alpaca-native"), ("mf", "mf")]
ALL_YEARS = list(range(1972, 2022))


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log((1.0 / (x + 1e-8)) - 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_varyear_glm(varyear_path: Path) -> tuple[float, float, int]:
    vy = pd.read_parquet(varyear_path).dropna(subset=["pred_logit_wavg", "obs_wavg"])
    X = sm.add_constant(vy["pred_logit_wavg"].to_numpy())
    y = vy["obs_wavg"].to_numpy()
    glm = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return float(glm.params[0]), float(glm.params[1]), int(len(vy))


def compute_pred_logit_wavg(
    wide_path: Path, weight_df: pd.DataFrame, target_vars: list[str]
) -> pd.DataFrame:
    cols = ["yearid_id"] + list(target_vars)
    wide = pd.read_parquet(wide_path, columns=cols)
    wide = wide.merge(weight_df, on="yearid_id", how="left")

    # Same rule as step5_long_to_varyear.py: models whose predictions leave
    # [0, 1] (MF emits raw ALS dot products) are aggregated on the
    # probability scale after clipping, not on the logit scale. The varyear
    # GLM fitted above was estimated on that same aggregate, so the covariate
    # here has to be built the same way.
    vals_all = wide[list(target_vars)].to_numpy(dtype=np.float64)
    clip_to_prob = bool(np.nanmin(vals_all) < 0.0 or np.nanmax(vals_all) > 1.0)
    if clip_to_prob:
        print(f"  predictions outside [0,1] (min {np.nanmin(vals_all):.3f}, "
              f"max {np.nanmax(vals_all):.3f}) - aggregating clipped probabilities")

    rows = []
    for year, idx in wide.groupby("year").groups.items():
        w = wide.loc[idx, "wtssall"].to_numpy()
        for var in target_vars:
            if var not in wide.columns:
                continue
            vals = wide.loc[idx, var].to_numpy(dtype=np.float64)
            logit_vals = np.clip(vals, 0.0, 1.0) if clip_to_prob else _logit(vals)
            mask = np.isfinite(logit_vals) & np.isfinite(w)
            if mask.sum() < 5:
                continue
            wm_logit = np.sum(w[mask] * logit_vals[mask]) / np.sum(w[mask])
            rows.append({
                "variable": var,
                "year": int(year),
                "pred_logit_wavg": float(wm_logit),
            })
    return pd.DataFrame(rows)


def compute_obs_bin(obs_df: pd.DataFrame, target_vars: list[str]) -> pd.DataFrame:
    sub = obs_df[obs_df["variable"].isin(target_vars)].dropna(
        subset=["binarized", "wtssall"])
    rows = []
    for (var, year), g in sub.groupby(["variable", "year"]):
        ds = DescrStatsW(g["binarized"].to_numpy(), weights=g["wtssall"].to_numpy())
        rows.append({
            "variable": var,
            "year": int(year),
            "mean": float(ds.mean),
            "pred_type": "obs_bin",
            "model": "alpaca",
        })
    return pd.DataFrame(rows)


def compute_ols(obs_tbl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, g in obs_tbl.groupby("variable"):
        g = g.dropna(subset=["mean"])
        if len(g) < 2:
            continue
        X = sm.add_constant(g["year"].to_numpy().astype(float))
        y = g["mean"].to_numpy()
        fit = sm.OLS(y, X).fit()
        Xp = sm.add_constant(np.array(ALL_YEARS, dtype=float))
        pred = np.clip(fit.predict(Xp), 0.0, 1.0)
        for yr, pr in zip(ALL_YEARS, pred):
            rows.append({
                "variable": var,
                "year": int(yr),
                "mean": float(pr),
                "pred_type": "ols",
                "model": "ols",
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading observations + weights ...")
    obs = pd.read_pickle(DATA / "df_analysis_slim.pkl").drop_duplicates(
        ["yearid_id", "question_id"])
    # 2021 GSS uses wtssnrps (non-response-adjusted) instead of wtssall.
    gss = pd.read_parquet(
        DATA / "gss_df_merged.parquet", columns=["year", "id", "wtssall", "wtssnrps"]
    )
    gss.loc[gss["year"] == 2021, "wtssall"] = gss.loc[gss["year"] == 2021, "wtssnrps"]
    gss = gss.drop(columns=["wtssnrps"])
    obs["id"] = obs["yearid"] % 10000
    obs = obs.merge(gss, on=["year", "id"], how="left")
    weight_df = obs[["yearid_id", "year", "wtssall"]].drop_duplicates("yearid_id")

    obs_tbl = compute_obs_bin(obs, TARGET_VARS)
    print(f"  obs_bin: {len(obs_tbl):,} (variable, year) pairs")

    parts = [obs_tbl]
    for mkey, mname in MODELS:
        print(f"\n=== {mkey} ===")
        vy_path = PRED / f"{mname}_partial_10_128_50__resample1_varyear.parquet"
        wide_path = PRED / f"{mname}_partial_10_128_50__resample1_wide.parquet"
        if not (vy_path.exists() and wide_path.exists()):
            print(f"  [skip] missing {vy_path.name} or {wide_path.name}")
            continue

        b0, b1, n_train = fit_varyear_glm(vy_path)
        print(f"  varyear GLM: β₀={b0:.4f}  β₁={b1:.4f}  (n={n_train:,})")

        logit_df = compute_pred_logit_wavg(wide_path, weight_df, TARGET_VARS)
        logit_df["mean"] = _sigmoid(b0 + b1 * logit_df["pred_logit_wavg"])
        logit_df["pred_type"] = "agg_calibrated"
        logit_df["model"] = mkey
        parts.append(logit_df[["variable", "year", "mean", "pred_type", "model"]])
        print(f"  agg_calibrated rows: {len(logit_df):,}")

    print("\n=== OLS baseline ===")
    ols_tbl = compute_ols(obs_tbl[["variable", "year", "mean"]])
    print(f"  OLS rows: {len(ols_tbl):,}")
    parts.append(ols_tbl)

    full = pd.concat(parts, ignore_index=True)
    full["lci"] = np.nan
    full["uci"] = np.nan

    pop_out = OUT / "counterfactual_pop_mean.parquet"
    full.to_parquet(pop_out, index=False)
    print(f"\nSaved: {pop_out} ({len(full):,} rows)")

    meta_src = pd.read_parquet(
        DATA / "gss_train_vars_corrected_binarized_again.parquet"
    )
    meta = (
        meta_src[meta_src["variable_name"].isin(TARGET_VARS)]
        [["variable_name", "variable_label", "question"]]
        .drop_duplicates("variable_name")
        .reset_index(drop=True)
    )
    meta_out = OUT / "counterfactual_variable_meta.parquet"
    meta.to_parquet(meta_out, index=False)
    print(f"Saved: {meta_out} ({len(meta)} variables)")


if __name__ == "__main__":
    main()
