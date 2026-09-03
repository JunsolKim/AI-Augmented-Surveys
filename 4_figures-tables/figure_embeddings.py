from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text


BASE = Path(__file__).resolve().parents[1]
IN_DIR = BASE / "data" / "fig_table_gen"
OUT_DIR = BASE / "output"

# Okabe-Ito colorblind-safe palette
PALETTE = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

# Past -> present gradient (viridis)
TIME_CMAP = "viridis"

PANEL_TITLE_SIZE = 22
LABEL_SIZE = 16


def _strip_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def _draw_question(ax: plt.Axes, df: pd.DataFrame) -> None:
    colors = [PALETTE[k % len(PALETTE)] for k in df["kmeans4"]]
    ax.scatter(df["x"], df["y"], c=colors, s=12.0, alpha=0.6, linewidths=0)
    ax.set_title("A. Question embeddings", loc="left", fontsize=PANEL_TITLE_SIZE)
    _strip_axes(ax)


def _draw_respondent(ax: plt.Axes, df: pd.DataFrame) -> None:
    palette10 = (PALETTE + PALETTE)[:10]
    colors = [palette10[k % len(palette10)] for k in df["kmeans10"]]
    ax.scatter(df["x"], df["y"], c=colors, s=2.5, alpha=0.5, linewidths=0)
    ax.set_title("B. Respondent embeddings", loc="left", fontsize=PANEL_TITLE_SIZE)
    _strip_axes(ax)


def _draw_period(ax: plt.Axes, df: pd.DataFrame) -> None:
    df = df.sort_values("year").reset_index(drop=True)

    x_span = df["x"].max() - df["x"].min()
    y_span = df["y"].max() - df["y"].min()
    pad_x = x_span * 0.18
    pad_y = y_span * 0.18
    ax.set_xlim(df["x"].min() - pad_x, df["x"].max() + pad_x)
    ax.set_ylim(df["y"].min() - pad_y, df["y"].max() + pad_y)

    ax.scatter(
        df["x"], df["y"], c=df["year"], cmap=TIME_CMAP,
        s=60, zorder=2, edgecolors="white", linewidths=0.6,
    )

    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    texts = [
        ax.text(
            row["x"], row["y"], str(int(row["year"])),
            fontsize=LABEL_SIZE, color="#222222",
            ha="center", va="center", zorder=3,
            path_effects=halo,
        )
        for _, row in df.iterrows()
    ]
    adjust_text(
        texts, ax=ax,
        expand=(1.4, 1.4),
        force_text=(0.6, 0.8),
        force_static=(0.4, 0.6),
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5, shrinkA=2, shrinkB=2),
    )

    ax.set_title("C. Period embeddings", loc="left", fontsize=PANEL_TITLE_SIZE)
    _strip_axes(ax)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    q = pd.read_parquet(IN_DIR / "embeddings_question.parquet")
    r = pd.read_parquet(IN_DIR / "embeddings_respondent.parquet")
    p = pd.read_parquet(IN_DIR / "embeddings_period.parquet")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    _draw_question(axes[0], q)
    _draw_respondent(axes[1], r)
    _draw_period(axes[2], p)
    fig.tight_layout()

    out_pdf = OUT_DIR / "figure_embeddings.pdf"
    out_png = OUT_DIR / "figure_embeddings.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
