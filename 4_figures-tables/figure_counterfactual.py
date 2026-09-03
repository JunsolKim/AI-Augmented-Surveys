from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from patsy import dmatrix


BASE = Path(__file__).resolve().parents[1]
IN_DIR = BASE / "data" / "fig_table_gen"
OUT_DIR = BASE / "output"

TARGET_VARS = ["marhomo1", "busing", "concong", "cohabit"]
VAR_LABELS = {
    "marhomo1": "homosexual couples have the right to marry",
    "busing": "favor the busing of Black and white\nschool children",
    "concong": "confidence in U.S. Congress",
    "cohabit": "lived with spouse before marriage",
}
PANEL_LETTERS = ["A", "B", "C", "D"]
MODEL = "alpaca"
PRED_TYPE = "agg_calibrated"
MARGIN = 0.03

PANEL_TITLE_SIZE = 12
AXIS_LABEL_SIZE = 13
TICK_SIZE = 12
LEGEND_SIZE = 11


def _classify(pred: float, obs: float) -> str:
    if np.isnan(obs):
        return "novel prediction"
    if abs(pred - obs) <= MARGIN:
        return "correct prediction"
    return "incorrect prediction"


def _panel_df(pop: pd.DataFrame, var: str) -> pd.DataFrame:
    pred = pop[
        (pop["variable"] == var)
        & (pop["model"] == MODEL)
        & (pop["pred_type"] == PRED_TYPE)
    ]
    obs = pop[
        (pop["variable"] == var)
        & (pop["model"] == "alpaca")
        & (pop["pred_type"] == "obs_bin")
    ][["year", "mean"]].rename(columns={"mean": "obs_mean"})

    df = pred.merge(obs, on="year", how="left").copy()
    df["overlap"] = [_classify(p, o) for p, o in zip(df["mean"], df["obs_mean"])]
    df["pred_prop"] = df["mean"]
    df["lci_prop"] = df["mean"] - MARGIN
    df["uci_prop"] = df["mean"] + MARGIN
    return df.sort_values("year").reset_index(drop=True)


def _obs_panel_df(pop: pd.DataFrame, var: str) -> pd.DataFrame:
    obs = pop[(pop["variable"] == var) & (pop["pred_type"] == "obs_bin")].copy()
    return obs[["year", "mean"]].rename(columns={"mean": "obs_prop"}).sort_values("year")


def _smooth_with_ci(
    years: np.ndarray, values: np.ndarray, df: int = 5, alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cubic regression spline (patsy `cr()`) + OLS → smoothed curve with
    95% CI band."""
    n = len(years)
    if n < 4:
        return years, values, values, values
    df_eff = min(df, max(3, n // 4))
    X = dmatrix(f"cr(x, df={df_eff})", {"x": years}, return_type="dataframe")
    fit = sm.OLS(values, X).fit()

    grid = np.linspace(years.min(), years.max(), 200)
    Xg = dmatrix(
        f"cr(x, df={df_eff})",
        {"x": grid},
        return_type="dataframe",
    )
    pr = fit.get_prediction(Xg).summary_frame(alpha=alpha)
    return grid, pr["mean"].to_numpy(), pr["mean_ci_lower"].to_numpy(), pr["mean_ci_upper"].to_numpy()


def _draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    obs_df: pd.DataFrame,
    roper_df: pd.DataFrame,
    title: str,
) -> tuple[float, float]:
    grid, sm_mean, sm_lo, sm_hi = _smooth_with_ci(
        df["year"].to_numpy(), df["pred_prop"].to_numpy()
    )
    ax.fill_between(grid, sm_lo, sm_hi, color="0.75", alpha=0.35, linewidth=0, zorder=1)
    ax.plot(grid, sm_mean, color="black", linewidth=1.0, zorder=2)

    groups = {
        "correct prediction":   dict(marker="o", mfc="black", mec="black", color="black"),
        "incorrect prediction": dict(marker="o", mfc="none",  mec="black", color="black"),
        "novel prediction":     dict(marker="*", mfc="blue",  mec="blue",  color="blue"),
    }
    for cat, style in groups.items():
        sub = df[df["overlap"] == cat]
        if sub.empty:
            continue
        ax.errorbar(
            sub["year"], sub["pred_prop"],
            yerr=[sub["pred_prop"] - sub["lci_prop"], sub["uci_prop"] - sub["pred_prop"]],
            fmt=style["marker"], markersize=4.6, markerfacecolor=style["mfc"],
            markeredgecolor=style["mec"], ecolor=style["color"],
            linewidth=0.6, capsize=0, zorder=3,
        )

    if not obs_df.empty:
        ax.scatter(
            obs_df["year"], obs_df["obs_prop"],
            marker="+", s=55, c="red", linewidths=1.3, zorder=4,
        )
    if not roper_df.empty:
        ax.scatter(
            roper_df["year"], roper_df["roper_mean"],
            marker="D", s=30, c="#E69F00", edgecolors="none", zorder=5,
        )

    ax.set_title(title, loc="left", fontsize=PANEL_TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, color="0.92", linewidth=0.5)

    y_vals = np.concatenate([
        df["pred_prop"].to_numpy(),
        df["lci_prop"].to_numpy(),
        df["uci_prop"].to_numpy(),
        obs_df["obs_prop"].to_numpy() if not obs_df.empty else np.array([]),
        roper_df["roper_mean"].to_numpy() if not roper_df.empty else np.array([]),
        sm_lo, sm_hi,
    ])
    y_vals = y_vals[np.isfinite(y_vals)]
    return (float(y_vals.min()), float(y_vals.max())) if len(y_vals) else (0.0, 1.0)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    pop = pd.read_parquet(IN_DIR / "counterfactual_pop_mean.parquet")
    roper = pd.read_parquet(IN_DIR / "counterfactual_roper.parquet")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), sharex=False)
    axes_flat = axes.flatten()

    panels = []
    for i, var in enumerate(TARGET_VARS):
        df = _panel_df(pop, var)
        obs_df = _obs_panel_df(pop, var)
        rdf = roper[roper["variable"] == var]
        title = f"{PANEL_LETTERS[i]}.  {VAR_LABELS[var]}"
        y_range = _draw_panel(axes_flat[i], df, obs_df, rdf, title)
        panels.append(y_range)
        # Y-label on left column only (panels 0 and 2)
        if i % 2 == 0:
            axes_flat[i].set_ylabel("Proportion", fontsize=AXIS_LABEL_SIZE)

    # Each panel: y-axis starts at 0, extends slightly above 0.5 (or higher
    # if the data exceeds that), with 0.1-spaced tick labels.
    for i, (_, hi) in enumerate(panels):
        upper = max(0.55, hi + 0.03)
        axes_flat[i].set_ylim(0.0, upper)
        axes_flat[i].yaxis.set_major_locator(MultipleLocator(0.1))

    handles = [
        Line2D([0], [0], marker="o", markerfacecolor="black", markeredgecolor="black",
               color="none", markersize=7, label="Correct prediction (±3%)"),
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="black",
               color="none", markersize=7, label="Incorrect prediction (±3%)"),
        Line2D([0], [0], marker="*", markerfacecolor="blue", markeredgecolor="blue",
               color="none", markersize=10, label="Novel prediction"),
        Line2D([0], [0], marker="+", color="red", linestyle="none",
               markersize=10, markeredgewidth=1.5, label="Observed (GSS)"),
        Line2D([0], [0], marker="D", markerfacecolor="#E69F00", markeredgecolor="#E69F00",
               color="none", markersize=7, label="Observed (Roper, orange)"),
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.995), ncol=3,
        frameon=False, fontsize=LEGEND_SIZE + 1,
        columnspacing=1.6, handletextpad=0.4,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_pdf = OUT_DIR / "figure_counterfactual.pdf"
    out_png = OUT_DIR / "figure_counterfactual.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
