from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_DIR = BASE / "data" / "fig_table_gen"
OUT_DIR = BASE / "output"

TASK_COLORS = {
    "Missing Data Imputation": "#B22222",     # firebrick (reddish brown)
    "Retrodiction": "#FF8C00",                # orange
    "Unasked Opinion Prediction": "#006400",  # darkgreen
}

TASKS = list(TASK_COLORS.keys())


def format_p(p: float) -> str:
    return "p < 2.2e-16" if p < 2.2e-16 else f"p = {p:.2e}"


def panel_roc(ax: plt.Axes) -> None:
    roc = pd.read_parquet(IN_DIR / "modelperformance_roc_downsampled.parquet")
    auc_df = pd.read_csv(IN_DIR / "modelperformance_roc_auc.csv").set_index("task")

    for task in TASKS:
        sub = roc[roc["task"] == task]
        auc = auc_df.loc[task, "auc"]
        label = f"{task} (AUC = {auc:.3f})"
        ax.plot(sub["fpr"], sub["tpr"], color=TASK_COLORS[task], linewidth=1.4, label=label)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("A. ROC Curve", loc="left", fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=12, frameon=False, handlelength=1.3)


def panel_scatter(ax: plt.Axes, task: str, panel_title: str) -> None:
    sc = pd.read_parquet(IN_DIR / "modelperformance_scatter.parquet")
    d = sc[(sc["task"] == task) & sc["pred_mean"].notna() & sc["obs_mean"].notna()].copy()

    # Pearson r and p-value.
    from scipy.stats import pearsonr
    r_val, p_val = pearsonr(d["pred_mean"], d["obs_mean"])
    pct3 = (d["diff_mean"].abs() < 0.03).mean() * 100
    pct5 = (d["diff_mean"].abs() < 0.05).mean() * 100
    mae  = d["diff_mean"].abs().mean()

    # Tiered alpha: |diff| < 0.03 → 0.8; < 0.05 → 0.4; else → 0.2.
    diff_abs = d["diff_mean"].abs()
    color = TASK_COLORS[task]
    for lo, hi, alpha in [(0.00, 0.03, 0.8), (0.03, 0.05, 0.4), (0.05, np.inf, 0.2)]:
        mask = (diff_abs >= lo) & (diff_abs < hi)
        if mask.any():
            ax.scatter(
                d.loc[mask, "pred_mean"], d.loc[mask, "obs_mean"],
                s=3, c=color, alpha=alpha, edgecolors="none",
            )

    # Annotations.
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
    ax.set_title(panel_title, loc="left", fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


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

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))

    panel_roc(axes[0, 0])
    panel_scatter(axes[0, 1], "Missing Data Imputation", "B. Missing Data Imputation")
    panel_scatter(axes[1, 0], "Retrodiction",            "C. Retrodiction")
    panel_scatter(axes[1, 1], "Unasked Opinion Prediction", "D. Unasked Opinion Prediction")

    for ax in axes.flat:
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout(w_pad=0.5, h_pad=0.5)

    out_pdf = OUT_DIR / "figure_modelperformance.pdf"
    out_png = OUT_DIR / "figure_modelperformance.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
