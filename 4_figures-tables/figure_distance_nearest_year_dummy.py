from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASE = Path(__file__).resolve().parents[1]
IN_PQ = BASE / "data" / "fig_table_gen" / "distance_nearest_year.parquet"
OUT_DIR = BASE / "output"

BIN_ORDER = ["1", "2-3", "4-5", "6-7", "8+"]

MODES = [
    ("retrodiction",  "A. Backcasting",   "#8B1A1A"),
    ("interpolation", "B. Interpolation", "#B86800"),
    ("forecast",      "C. Forecasting",   "#1A5F1A"),
]


def bin_distance(d):
    if pd.isna(d):
        return np.nan
    d = int(d)
    if d <= 1:    return "1"
    if d <= 3:    return "2-3"
    if d <= 5:    return "4-5"
    if d <= 7:    return "6-7"
    return "8+"


def within_bin_metrics(d_mode):
    """Return DataFrame with one row per distance bin: rho, mae, n."""
    rows = []
    for label in BIN_ORDER:
        sub = d_mode[d_mode["dist_bin"] == label]
        n = len(sub)
        if n == 0:
            continue
        if n >= 3 and sub["GLM_predicted"].std() > 0 and sub["obs_wavg"].std() > 0:
            rho, _ = spearmanr(sub["GLM_predicted"], sub["obs_wavg"])
        else:
            rho = np.nan
        mae = float(sub["abs_error"].mean())
        mae_se = (float(sub["abs_error"].std(ddof=1) / np.sqrt(n))
                  if n >= 2 else 0.0)
        rows.append({"dist_bin": label, "rho": rho, "mae": mae,
                     "mae_se": mae_se, "n": n})
    return pd.DataFrame(rows)


def panel_rho(ax, m, color, title=None, show_ylabel=False):
    if m.empty:
        ax.text(0.5, 0.5, "n=0", transform=ax.transAxes, ha="center", va="center")
        return
    xs = np.arange(len(m))
    ax.plot(xs, m["rho"], "-o", color=color, linewidth=1.5, markersize=5)
    for x, y, n in zip(xs, m["rho"], m["n"]):
        if pd.notna(y):
            ax.text(x, y - 0.012, f"n={n}", ha="center", va="top", fontsize=13)
    ax.set_xticks(xs)
    ax.set_xticklabels(m["dist_bin"])
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("Distance to nearest training year")
    if show_ylabel:
        ax.set_ylabel("Spearman(pred, obs)")
    if title is not None:
        ax.set_title(title, loc="left", fontsize=18)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(IN_PQ)
    df = df.dropna(subset=["distance_to_nearest", "abs_error",
                           "GLM_predicted", "obs_wavg"]).copy()
    df["dist_bin"] = df["distance_to_nearest"].map(bin_distance)
    df["dist_bin"] = pd.Categorical(df["dist_bin"], categories=BIN_ORDER,
                                    ordered=True)

    print("Per-bin metrics by mode:")
    for mode, _, _ in MODES:
        m = within_bin_metrics(df[df["mode"] == mode])
        print(f"\n[{mode}]")
        print(m.to_string(index=False))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "axes.titlesize": 26,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "0.92",
        "grid.linewidth": 0.5,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for j, (mode, title, color) in enumerate(MODES):
        m = within_bin_metrics(df[df["mode"] == mode])
        panel_rho(axes[j], m, color, title=title, show_ylabel=(j == 0))
    fig.tight_layout()

    out_pdf = OUT_DIR / "figure_distance_nearest_year_dummy.pdf"
    out_png = OUT_DIR / "figure_distance_nearest_year_dummy.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"\nsaved: {out_pdf}")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
