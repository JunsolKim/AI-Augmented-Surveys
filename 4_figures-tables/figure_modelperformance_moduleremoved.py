from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.weightstats import DescrStatsW


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
CACHE = DATA / "fig_table_gen" / "modelperformance_moduleremoved_scatter.parquet"
OUT = BASE / "output"
STEM = "figure_modelperformance_moduleremoved"

MODEL = "maicomputer_alpaca-native"
COLOR = "#0E8A8A"  # teal — distinct from the 3-task palette in figure_modelperformance.


def compute_agg() -> pd.DataFrame:
    df = pd.read_pickle(DATA / "df_analysis_slim.pkl")
    df = df.drop_duplicates(["yearid_id", "question_id"])

    gss = pd.read_parquet(DATA / "gss_df_merged.parquet", columns=["year", "id", "wtssall"])
    df["id"] = df["yearid"] % 10000
    df = df.merge(gss, on=["year", "id"], how="left")

    module_year_counts = df.groupby("module")["year"].nunique()
    multi_year_modules = set(module_year_counts[module_year_counts > 1].index)
    print(f"multi-year modules: {len(multi_year_modules)}")

    df_obs = df[[
        "yearid_id", "question_id", "variable", "year",
        "module", "binarized", "wtssall",
    ]].copy()

    wide_path = PRED / f"{MODEL}_module_10_128_50__resample1_wide.parquet"
    wide = pd.read_parquet(wide_path)
    glm_cols = [c for c in wide.columns if c.endswith("_rescale_logit_glm")]

    long = wide[["yearid_id"] + glm_cols].melt(
        id_vars="yearid_id", var_name="variable", value_name="predicted",
    )
    long["variable"] = long["variable"].str.replace("_rescale_logit_glm", "", regex=False)

    val = (
        long.merge(df_obs, on=["yearid_id", "variable"], how="inner")
            .dropna(subset=["binarized", "predicted", "wtssall"])
    )
    val = val[val["module"].isin(multi_year_modules)]
    print(f"row-level merged: {len(val):,} rows, {val['variable'].nunique()} variables")

    rows = []
    for (var, yr), grp in val.groupby(["variable", "year"]):
        if len(grp) < 5:
            continue
        w = grp["wtssall"].values
        rows.append({
            "variable": var,
            "year": yr,
            "pred_mean": DescrStatsW(grp["predicted"].values, weights=w).mean,
            "obs_mean":  DescrStatsW(grp["binarized"].values, weights=w).mean,
        })
    agg = pd.DataFrame(rows)
    agg["diff_mean"] = agg["pred_mean"] - agg["obs_mean"]
    return agg


def load_or_compute() -> pd.DataFrame:
    if CACHE.exists():
        print(f"loading cache: {CACHE}")
        return pd.read_parquet(CACHE)
    print("computing aggregate from raw predictions ...")
    agg = compute_agg()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(CACHE)
    print(f"cached: {CACHE}  ({len(agg)} (var,year) pairs)")
    return agg


def format_p(p: float) -> str:
    return "p < 2.2e-16" if p < 2.2e-16 else f"p = {p:.2e}"


def plot(agg: pd.DataFrame) -> None:
    d = agg.dropna(subset=["pred_mean", "obs_mean"]).copy()
    d["diff_mean"] = d["pred_mean"] - d["obs_mean"]

    r_val, p_val = pearsonr(d["pred_mean"], d["obs_mean"])
    pct3 = (d["diff_mean"].abs() < 0.03).mean() * 100
    pct5 = (d["diff_mean"].abs() < 0.05).mean() * 100
    mae  = d["diff_mean"].abs().mean()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 16,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "0.92",
        "grid.linewidth": 0.5,
    })

    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    diff_abs = d["diff_mean"].abs()
    for lo, hi, alpha in [(0.00, 0.03, 0.8), (0.03, 0.05, 0.4), (0.05, np.inf, 0.2)]:
        mask = (diff_abs >= lo) & (diff_abs < hi)
        if mask.any():
            ax.scatter(
                d.loc[mask, "pred_mean"], d.loc[mask, "obs_mean"],
                s=4, c=COLOR, alpha=alpha, edgecolors="none",
            )

    ax.text(0.02, 0.95, f"R = {r_val:.2f}, {format_p(p_val)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.text(0.02, 0.88, f"% Correct (±3%) = {pct3:.1f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=12.5)
    ax.text(0.02, 0.81, f"% Correct (±5%) = {pct5:.1f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=12.5)
    ax.text(0.02, 0.74, f"MAE = {mae:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=12.5)

    ax.set_xlabel("Predicted Mean Responses")
    ax.set_ylabel("Observed Mean Responses")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    OUT.mkdir(exist_ok=True)
    out_png = OUT / f"{STEM}.png"
    out_pdf = OUT / f"{STEM}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")
    print(f"N={len(d)}  R_pearson={r_val:.4f}  MAE={mae:.4f}  ±3%={pct3:.1f}%  ±5%={pct5:.1f}%")


def main() -> None:
    agg = load_or_compute()
    plot(agg)


if __name__ == "__main__":
    main()
