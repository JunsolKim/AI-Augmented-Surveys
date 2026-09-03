from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


REPO = Path(__file__).resolve().parents[1]
PARQUET = REPO / "data" / "gss_train_vars_corrected_binarized_again.parquet"
OUT_DIR = REPO / "output"


# ---------- load counts from data where possible ----------
df = pd.read_parquet(PARQUET)
stage3_n = len(df)
stage4_n = int((df["use_variable"] == 1).sum())
stage5_n = int((df["use_variable_final"] == 1).sum())

counts = [
    "7,735 variables",      # Stage 1: GSS Data Explorer API snapshot
    "5,179 variables",      # Stage 2
    f"{stage3_n:,} variables",
    f"{stage4_n:,} variables",
    f"{stage5_n:,} variables",
]


# ---------- font sizes ----------
COUNT_FS = 56
STAGE_FS = 46
ANNOT_FS = 40


# ---------- stage labels & annotations ----------
stage_labels = [
    "Stage 1:\nCollect candidate GSS\nvariables available in\nthe GSS Data Explorer",
    "Stage 2:\nKeep discrete categorical\nvariables with two or more\nvalid response labels",
    "Stage 3:\nAnnotate each variable\nwith a question type and\na binarizability flag using\na large language model",
    "Stage 4:\nKeep substantive question\ntypes that are binarizable",
    "Stage 5:\nHuman review and\ncorrection of the LLM\nannotations",
]

annotations = {
    1: (
        "The LLM assigns each variable to one of nine question\n"
        "types: Behavioral, Attitudinal/Opinion, Demographic,\n"
        "Objective/Knowledge/Factual, Questions about Other\n"
        "People, Open-ended, Derived/Computed, Metadata/\n"
        "Paradata, and Miscellaneous. It also decides whether\n"
        "the response options admit a meaningful 1/0 contrast.\n"
        "Ordinal and Likert scales are allowed; purely nominal\n"
        "and continuous items are rejected. 'Don't know' and\n"
        "'refused' are treated as missing; neutral midpoints\n"
        "are mapped to 0."
    ),
    2: (
        "Keep only Attitudinal/Opinion, Behavioral, and\n"
        "Objective/Knowledge/Factual questions that are also\n"
        "flagged as binarizable. Drop the six remaining\n"
        "question types and any non-binarizable variables."
    ),
    3: (
        "A human reviewer audits the LLM category and\n"
        "binarization labels and overrides borderline cases."
    ),
}


# ---------- layout constants (inches) ----------
BOX_W, BOX_H = 7.6, 2.6

ANNOT_LINE_H = 0.80
ANNOT_CHAR_W = 0.32
ANNOT_PAD_X = 0.55
ANNOT_PAD_Y = 0.60

LEFT_MARGIN = 10.0
LEFT_X = 0.4

max_note_chars = max(
    max(len(l) for l in note.split("\n")) for note in annotations.values()
)
NOTE_W = ANNOT_CHAR_W * max_note_chars + 2 * ANNOT_PAD_X

CX = LEFT_MARGIN + BOX_W / 2
RIGHT_X = CX + BOX_W / 2 + 1.0

GAP = 5.6
TOP_PAD = 0.8
BOTTOM_PAD = 0.8
N = len(counts)

FIG_W = LEFT_MARGIN + BOX_W + 1.0 + NOTE_W + 0.6
FIG_H = TOP_PAD + BOTTOM_PAD + (N - 1) * GAP + BOX_H


# ---------- figure ----------
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")

TOP_Y = FIG_H - TOP_PAD - BOX_H / 2
box_centers_y = [TOP_Y - i * GAP for i in range(N)]

for cy, txt in zip(box_centers_y, counts):
    ax.add_patch(Rectangle(
        (CX - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        linewidth=1.8, edgecolor="black", facecolor="white",
    ))
    ax.text(CX, cy, txt, ha="center", va="center", fontsize=COUNT_FS)

arrow_segments = []
for i in range(N - 1):
    y_tail = box_centers_y[i] - BOX_H / 2
    y_head = box_centers_y[i + 1] + BOX_H / 2
    ax.add_patch(FancyArrowPatch(
        (CX, y_tail), (CX, y_head),
        arrowstyle="-|>", mutation_scale=34,
        linewidth=1.8, color="black", shrinkA=0, shrinkB=0,
    ))
    arrow_segments.append(((CX, y_tail), (CX, y_head)))

for cy, lbl in zip(box_centers_y, stage_labels):
    ax.text(
        LEFT_X, cy + BOX_H / 2 + 0.1, lbl,
        ha="left", va="top",
        fontsize=STAGE_FS, linespacing=1.3,
    )

for arrow_idx, note in annotations.items():
    (_, y_tail), (_, y_head) = arrow_segments[arrow_idx]
    mid_y = 0.5 * (y_tail + y_head)

    n_lines = len(note.split("\n"))
    box_h = ANNOT_LINE_H * n_lines + 2 * ANNOT_PAD_Y
    bx = RIGHT_X
    by = mid_y - box_h / 2

    ax.add_patch(Rectangle(
        (bx, by), NOTE_W, box_h,
        linewidth=1.3, edgecolor="0.35", facecolor="white",
    ))
    ax.text(
        bx + ANNOT_PAD_X, by + box_h - ANNOT_PAD_Y, note,
        ha="left", va="top",
        fontsize=ANNOT_FS, linespacing=1.3,
    )
    ax.add_patch(FancyArrowPatch(
        (bx, mid_y), (CX + 0.05, mid_y),
        arrowstyle="->", mutation_scale=26,
        linewidth=1.3, color="0.55", shrinkA=0, shrinkB=2,
    ))


# ---------- save ----------
OUT_DIR.mkdir(exist_ok=True)
out_pdf = OUT_DIR / "figure_varselection.pdf"
out_png = OUT_DIR / "figure_varselection.png"
fig.savefig(out_pdf, pad_inches=0.3)
fig.savefig(out_png, pad_inches=0.3, dpi=150)
print(f"FIG_W x FIG_H = {FIG_W:.2f} x {FIG_H:.2f} in")
print(f"saved: {out_pdf}")
print(f"saved: {out_png}")
