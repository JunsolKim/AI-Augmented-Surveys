from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_PATH = BASE / "data" / "fig_table_gen" / "years_vs_auc.parquet"
OUT_DIR = BASE / "output"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(IN_PATH).sort_values("n_year")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 16,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "0.92",
        "grid.linewidth": 0.5,
    })

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["n_year"], df["auc"], color="black", linewidth=0.8, zorder=2)
    ax.scatter(df["n_year"], df["auc"], s=18, color="black", zorder=3)

    ax.set_xlabel("Number of years per question")
    ax.set_ylabel("AUC")
    x_min, x_max = int(df["n_year"].min()), int(df["n_year"].max())
    ax.set_xticks(range(x_min, x_max + 1, 5))
    ax.set_ylim(0, 1)

    fig.tight_layout()
    out_pdf = OUT_DIR / "figure_years_vs_auc.pdf"
    out_png = OUT_DIR / "figure_years_vs_auc.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
