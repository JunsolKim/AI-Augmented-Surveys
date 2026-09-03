from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASE = Path(__file__).resolve().parents[1]
IN_PATH = BASE / "data" / "fig_table_gen" / "polviews_rank.parquet"
OUT_DIR = BASE / "output"
STEM = "figure_polviews_rank_scatter"

COLOR = "#0E8A8A"  # teal, same hue family as the moduleremoved scatter.


def format_p(p: float) -> str:
    return "p < 2.2e-16" if p < 2.2e-16 else f"p = {p:.2e}"


def panel_scatter(ax: plt.Axes) -> None:
    df = pd.read_parquet(IN_PATH)
    d = df.dropna(subset=["spearman_rho", "alpaca_partial_auc"]).copy()
    d["abs_rho"] = d["spearman_rho"].abs()

    r_abs, p_abs = spearmanr(d["abs_rho"], d["alpaca_partial_auc"])
    n = len(d)

    # Tiered alpha by |rho|: more ideologically-aligned variables stand out.
    for lo, hi, alpha in [(0.30, np.inf, 0.85),
                          (0.15, 0.30,   0.45),
                          (0.00, 0.15,   0.18)]:
        mask = (d["abs_rho"] >= lo) & (d["abs_rho"] < hi)
        if mask.any():
            ax.scatter(
                d.loc[mask, "abs_rho"],
                d.loc[mask, "alpaca_partial_auc"],
                s=6, c=COLOR, alpha=alpha, edgecolors="none",
            )

    # Reference line: chance AUC.
    ax.axhline(0.5, color="0.55", linewidth=0.6, linestyle="--", zorder=0)

    # Annotation.
    ax.text(0.02, 0.97, f"ρ = {r_abs:.2f}, {format_p(p_abs)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=13)

    ax.set_xlabel(r"$|\rho(\mathrm{polviews})|$")
    ax.set_ylabel("Alpaca AUC")
    ax.set_xlim(0.0, 0.6)
    ax.set_ylim(0.3, 1.0)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

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

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    panel_scatter(ax)
    fig.tight_layout()

    out_pdf = OUT_DIR / f"{STEM}.pdf"
    out_png = OUT_DIR / f"{STEM}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
