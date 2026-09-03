from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_PATH = BASE / "data" / "fig_table_gen" / "auc_by_nq.parquet"
OUT_DIR = BASE / "output"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(IN_PATH)

    baseline = float(df.loc[df["max_q"] == "all", "auc"].iloc[0])
    pts = df[df["max_q"] != "all"].copy()
    pts["max_q"] = pts["max_q"].astype(int)
    pts = pts.sort_values("max_q")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 17,
        "figure.titlesize": 22,
        "axes.axisbelow": True,
    })

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(pts["max_q"], pts["auc"], color="black", linewidth=1.2, zorder=2)
    ax.scatter(pts["max_q"], pts["auc"], s=28, color="black", zorder=3)

    ax.axhline(baseline, linestyle=":", color="0.5", linewidth=1.0,
               label=f"Baseline AUC = {baseline:.3f}")

    ax.set_xlabel("Number of survey questions in training data")
    ax.set_ylabel("AUC")

    ax.yaxis.grid(True, color="0.9", linewidth=0.6)
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_color("0.2")

    ax.legend(loc="lower right", frameon=True, fancybox=False,
              edgecolor="0.7", framealpha=1.0)

    fig.tight_layout()
    out_pdf = OUT_DIR / "figure_n_questions_vs_auc.pdf"
    out_png = OUT_DIR / "figure_n_questions_vs_auc.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")
    print(f"baseline AUC = {baseline:.6f}; n points = {len(pts)}")


if __name__ == "__main__":
    main()
