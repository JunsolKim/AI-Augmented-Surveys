from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "output"

GRAY_DOT = "#4D4D4D"
HELD_OUT = "#D7263D"
BAND = "#E5E5E5"
ANNOT = "#888888"

EXAMPLES = [
    {
        "label": "Variable A",
        "regime": "Backcasting",
        "train_years": [2008, 2010, 2012, 2014, 2018, 2022],
        "heldout": 1990,
        "note": "held-out year before the training window",
    },
    {
        "label": "Variable B",
        "regime": "Interpolation",
        "train_years": [1990, 1996, 2000, 2008, 2014, 2018],
        "heldout": 2004,
        "note": "held-out year inside the training window",
    },
    {
        "label": "Variable C",
        "regime": "Forecasting",
        "train_years": [1972, 1980, 1988, 1994, 2000],
        "heldout": 2018,
        "note": "held-out year after the training window",
    },
]

X_MIN, X_MAX = 1970, 2024


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    })

    fig, ax = plt.subplots(figsize=(9.0, 4.4))

    n = len(EXAMPLES)
    y_positions = list(range(n, 0, -1))  # 3, 2, 1 (top to bottom)

    for y, ex in zip(y_positions, EXAMPLES):
        train = ex["train_years"]
        held = ex["heldout"]
        lo, hi = min(train), max(train)

        # Training window band
        ax.add_patch(Rectangle(
            (lo, y - 0.28), hi - lo, 0.56,
            facecolor=BAND, edgecolor="none", zorder=1,
        ))

        # Baseline (timeline) line
        ax.hlines(y, X_MIN + 1, X_MAX - 1,
                  color="#BBBBBB", linewidth=0.7, zorder=2)

        # Training observations
        ax.scatter(train, [y] * len(train),
                   marker="o", s=70, color=GRAY_DOT, zorder=3,
                   edgecolors="white", linewidths=0.8)

        # Held-out year
        ax.scatter([held], [y],
                   marker="X", s=210, color=HELD_OUT, zorder=4,
                   edgecolors="white", linewidths=1.2,
                   label="Held-out cell" if y == y_positions[0] else None)

        # Variable label (left)
        ax.text(X_MIN - 0.5, y, ex["label"],
                ha="right", va="center", fontsize=15)

        # Regime label (right)
        ax.text(X_MAX + 0.5, y, ex["regime"],
                ha="left", va="center", fontsize=15,
                fontweight="bold", color="#222222")

    # X-axis: years
    ax.set_xlim(X_MIN - 6, X_MAX + 13)
    ax.set_ylim(0.4, n + 0.8)
    ax.set_yticks([])
    decade_ticks = list(range(1970, 2025, 10))
    ax.set_xticks(decade_ticks)
    ax.set_xticklabels([str(y) for y in decade_ticks], fontsize=14)
    ax.tick_params(axis="x", length=3)
    ax.set_xlabel("GSS year", fontsize=16)

    # Legend (top, 2 rows)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="white",
                   markerfacecolor=GRAY_DOT, markeredgecolor="white",
                   markersize=12, label="Observation in training data"),
        plt.Line2D([0], [0], marker="X", color="white",
                   markerfacecolor=HELD_OUT, markeredgecolor="white",
                   markersize=16, label="Held-out year to be predicted"),
        Rectangle((0, 0), 1, 1, facecolor=BAND, edgecolor="none",
                  label="Training window for this variable"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        ncol=2,
        frameon=False,
        fontsize=15,
        handletextpad=0.6,
        columnspacing=1.6,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.85])

    out_pdf = OUT / "figure_regime_schematic.pdf"
    out_png = OUT / "figure_regime_schematic.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
