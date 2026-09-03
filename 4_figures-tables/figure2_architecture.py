from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Arc, Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle, Wedge, PathPatch)
from matplotlib.path import Path as MPath

# ------------------------ palette ------------------------
RED        = '#C62828'
BLUE       = '#1565C0'
ORANGE     = '#F57C00'
TEAL       = '#00897B'
PURPLE     = '#6A1B9A'
PANEL_B_BG = '#F1ECE0'
DASH_GREY  = '#9E9E9E'
EDGE       = '#222222'

OUT_DIR = Path(__file__).resolve().parents[1] / 'output'


# ------------------------ helpers ------------------------
def box(ax, cx, cy, w, h, text, *, fc='white', ec=EDGE, fontsize=9,
        text_color='black', lw=1.2, fontweight='normal'):
    bx = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle='round,pad=0.3,rounding_size=0.6',
                        fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(bx)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight=fontweight,
            zorder=4)


def arrow(ax, x0, y0, x1, y1, *, color=EDGE, lw=1.3, mut=14, ls='-'):
    a = FancyArrowPatch((x0, y0), (x1, y1),
                        arrowstyle='-|>', mutation_scale=mut,
                        lw=lw, color=color, linestyle=ls, zorder=2,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def person(ax, cx, cy, color, *, scale=1.0, label=None, lab_dx=4.0,
           label_fontsize=10.5):
    """Gender-neutral icon: small circle head sitting directly on top of a
    rounded-rectangle torso of uniform width (no shoulders, no hair, no
    clothing, no face). Slim balanced proportions."""
    head_r = 1.05 * scale
    torso_w, torso_h = 1.7 * scale, 2.2 * scale
    torso_bottom = cy - torso_h / 2 - 0.35 * scale
    torso = FancyBboxPatch((cx - torso_w / 2, torso_bottom), torso_w, torso_h,
                           boxstyle='round,pad=0,rounding_size=0.5',
                           fc=color, ec='none', zorder=3)
    ax.add_patch(torso)
    head_y = torso_bottom + torso_h + head_r * 0.7
    ax.add_patch(Circle((cx, head_y), head_r, fc=color, ec='none', zorder=3))
    if label:
        ax.text(cx + lab_dx, cy, label, ha='left', va='center',
                color=color, fontsize=label_fontsize, fontweight='bold')


def vector_bar(ax, left_x, cy, n_cells, base_color, *,
               cell_w=1.4, cell_h=2.6, seed=0):
    """Horizontal bar of discrete cells (one row of small squares).
    Returns (left_x, right_x, cy)."""
    rng = np.random.default_rng(seed)
    shades = 0.32 + 0.62 * rng.random(n_cells)
    for i in range(n_cells):
        x = left_x + i * cell_w
        rect = Rectangle((x, cy - cell_h / 2), cell_w * 0.92, cell_h,
                         fc=base_color, alpha=shades[i],
                         ec=EDGE, lw=0.4, zorder=3)
        ax.add_patch(rect)
    return left_x, left_x + n_cells * cell_w, cy


def padlock(ax, cx, cy, *, scale=1.0, color=EDGE):
    body_w, body_h = 1.0 * scale, 0.85 * scale
    ax.add_patch(Rectangle((cx - body_w / 2, cy - body_h / 2),
                           body_w, body_h, fc=color, ec='none', zorder=4))
    ax.add_patch(Arc((cx, cy + body_h / 2),
                     body_w * 0.75, body_w * 0.75,
                     theta1=0, theta2=180,
                     color=color, lw=1.5 * scale, zorder=4))
    ax.add_patch(Circle((cx, cy), 0.13 * scale, fc='white', ec='none',
                        zorder=5))


def magnifier(ax, cx, cy, *, scale=1.0, color=EDGE):
    r = 0.85 * scale
    ax.add_patch(Circle((cx, cy), r, fc='none', ec=color, lw=1.5, zorder=4))
    ax.plot([cx + r * 0.7, cx + r * 1.7], [cy - r * 0.7, cy - r * 1.7],
            color=color, lw=1.5, zorder=4, solid_capstyle='round')


def right_brace(ax, x, y_top, y_bot, *, depth=1.6, lw=1.2, color=EDGE):
    """A right-pointing curly brace (} shape) covering [y_bot, y_top] at x,
    with the tip at x+depth, mid-y."""
    mid = (y_top + y_bot) / 2
    p = MPath([
        (x,           y_top),
        (x + depth*0.5, y_top),
        (x + depth*0.5, mid + 0.5),
        (x + depth,   mid),
        (x + depth*0.5, mid - 0.5),
        (x + depth*0.5, y_bot),
        (x,           y_bot),
    ], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3,
        MPath.LINETO,
        MPath.CURVE3, MPath.CURVE3, MPath.LINETO])
    ax.add_patch(PathPatch(p, fc='none', ec=color, lw=lw, zorder=2))
    return x + depth, mid


# ------------------------ figure ------------------------
def main():
    fig = plt.figure(figsize=(16.5, 10.5))
    plt.rcParams['font.family'] = 'DejaVu Sans'

    ax_A = fig.add_axes([0.01, 0.56, 0.98, 0.42])
    ax_B = fig.add_axes([0.01, 0.02, 0.98, 0.52])
    for ax in (ax_A, ax_B):
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    ax_B.add_patch(Rectangle((0, 0), 100, 100, fc=PANEL_B_BG,
                             ec='none', zorder=-10))

    ax_A.text(1.5, 95, 'A', fontsize=24, fontweight='bold')
    ax_B.text(1.5, 95, 'B', fontsize=24, fontweight='bold')

    # ===================================================
    # PANEL A
    # ===================================================
    in_cx, in_w = 11, 18
    box(ax_A, in_cx, 78, in_w, 26,
        'Survey question (text)\n\n"Do you agree or disagree\nthat homosexual couples\nshould have the right\nto marry one another?"',
        fontsize=8.4)
    box(ax_A, in_cx, 48, in_w, 9,
        'Respondent ID\n(e.g., #12345)', fontsize=9)
    box(ax_A, in_cx, 30, in_w, 9,
        'Survey year\n(e.g., 1995)', fontsize=9)

    # Model box
    mdl_cx, mdl_cy, mdl_w, mdl_h = 42, 54, 24, 20
    box(ax_A, mdl_cx, mdl_cy, mdl_w, mdl_h,
        'Individual-level\nPrediction Model',
        fontsize=11.5, fontweight='bold')
    ax_A.text(mdl_cx, mdl_cy - 5, '(see Panel B)', ha='center', va='center',
              fontsize=9, fontstyle='italic')
    magnifier(ax_A, mdl_cx + mdl_w / 2 + 1.5, mdl_cy + mdl_h / 2 - 1.5,
              scale=1.6)

    for y in (78, 48, 30):
        arrow(ax_A, in_cx + in_w / 2 + 0.2, y, mdl_cx - mdl_w / 2 - 0.2,
              mdl_cy + (y - 54) * 0.42)

    pred_cx = 72
    ax_A.text(pred_cx, 90, 'Individual-level\npredictions',
              ha='center', va='top', fontsize=10, fontweight='bold')
    person(ax_A, pred_cx, 73, RED, scale=1.5, label='Agree')
    person(ax_A, pred_cx, 60, RED, scale=1.5, label='Agree')
    ax_A.text(pred_cx, 50.5, '⋮', ha='center', va='center', fontsize=20,
              fontweight='bold')
    person(ax_A, pred_cx, 38, BLUE, scale=1.5, label='Disagree')

    arrow(ax_A, mdl_cx + mdl_w / 2 + 0.2, mdl_cy, pred_cx - 4, mdl_cy)

    # Pie chart
    pie_cx, pie_cy, pie_r = 91, 54, 9
    ax_A.add_patch(Wedge((pie_cx, pie_cy), pie_r, -126, 90,
                         fc=RED, ec='white', lw=1.5, zorder=2))
    ax_A.add_patch(Wedge((pie_cx, pie_cy), pie_r, 90, 234,
                         fc=BLUE, ec='white', lw=1.5, zorder=2))
    ax_A.text(pie_cx + 2.6, pie_cy - 1, 'Agree', color='white',
              ha='center', va='center', fontsize=10, fontweight='bold',
              zorder=3)
    ax_A.text(pie_cx - 3.0, pie_cy + 2, 'Disagree', color='white',
              ha='center', va='center', fontsize=8.5, fontweight='bold',
              zorder=3)
    ax_A.text(pie_cx, 80, 'Population-level\naggregation\n(weighted by\nsurvey weights)',
              ha='center', va='top', fontsize=9)

    arrow(ax_A, pred_cx + 5, mdl_cy, pie_cx - pie_r - 1, pie_cy)

    # Zoom-in dashed lines
    p_left  = ax_A.transData.transform((mdl_cx - mdl_w / 2, mdl_cy - mdl_h / 2))
    p_right = ax_A.transData.transform((mdl_cx + mdl_w / 2, mdl_cy - mdl_h / 2))
    pB_l = ax_B.transData.transform((1.5, 99))
    pB_r = ax_B.transData.transform((98.5, 99))
    inv = fig.transFigure.inverted()
    p_left_f, p_right_f = inv.transform(p_left), inv.transform(p_right)
    pB_l_f, pB_r_f = inv.transform(pB_l), inv.transform(pB_r)
    fig.lines.extend([
        plt.Line2D([p_left_f[0], pB_l_f[0]], [p_left_f[1], pB_l_f[1]],
                   transform=fig.transFigure, color=DASH_GREY,
                   lw=0.9, ls='--', zorder=0),
        plt.Line2D([p_right_f[0], pB_r_f[0]], [p_right_f[1], pB_r_f[1]],
                   transform=fig.transFigure, color=DASH_GREY,
                   lw=0.9, ls='--', zorder=0),
    ])

    # ===================================================
    # PANEL B (horizontal embedding bars)
    # ===================================================
    ax_B.text(50, 96,
              'Model architecture (zoom-in of the Individual-level Prediction Model)',
              ha='center', va='top', fontsize=11.5, fontweight='bold',
              fontstyle='italic')

    # Layout zones (tight, leaves room on right for cross/dense/output)
    inp_cx   = 7.5
    llm_cx   = 22
    lin_cx   = 36
    bar_left = 44              # left edge of each embedding bar
    bar_n    = 14
    bar_cw   = 1.1
    bar_ch   = 2.4
    bar_right = bar_left + bar_n * bar_cw   # ~59.4

    R1_y = 84
    R2_y = 64
    R3_y = 44

    # ---- Row 1 (Question) ----
    box(ax_B, inp_cx, R1_y, 12, 9,
        'Survey question\ntext\n(gay-marriage)', fontsize=8.4)
    arrow(ax_B, inp_cx + 6, R1_y, llm_cx - 6.5, R1_y)
    box(ax_B, llm_cx, R1_y, 13, 9,
        'Pre-trained LLM\n(e.g., Alpaca-7b,\nfrozen)', fontsize=8.4)
    padlock(ax_B, llm_cx + 5.0, R1_y + 3.0, scale=1.3)
    ax_B.text(llm_cx, R1_y - 7.0,
              "output: last layer's\nresidual stream",
              ha='center', va='top', fontsize=7.6, fontstyle='italic')
    arrow(ax_B, llm_cx + 6.5, R1_y, lin_cx - 5.5, R1_y)
    box(ax_B, lin_cx, R1_y, 11, 7,
        'Trainable\nlinear layer', fontsize=8.4)
    arrow(ax_B, lin_cx + 5.5, R1_y, bar_left - 0.5, R1_y)
    qL, qR, _ = vector_bar(ax_B, bar_left, R1_y, bar_n, ORANGE,
                            cell_w=bar_cw, cell_h=bar_ch, seed=11)
    ax_B.text(qL, R1_y + bar_ch / 2 + 1.4, 'Question embedding (50-d)',
              ha='left', va='bottom', fontsize=9, color=ORANGE,
              fontweight='bold')
    ax_B.text((qL + qR) / 2, R1_y - bar_ch / 2 - 1.4,
              'semantic meaning of the question;\nfine-tuned from LLM output',
              ha='center', va='top', fontsize=7.2, color='black')

    # ---- Row 2 (Respondent) ----
    box(ax_B, inp_cx, R2_y, 12, 7,
        'Respondent ID\n(#12345)', fontsize=8.6)
    arrow(ax_B, inp_cx + 6, R2_y, bar_left - 0.5, R2_y)
    rL, rR, _ = vector_bar(ax_B, bar_left, R2_y, bar_n, TEAL,
                            cell_w=bar_cw, cell_h=bar_ch, seed=22)
    ax_B.text(rL, R2_y + bar_ch / 2 + 1.4, 'Respondent embedding (50-d)',
              ha='left', va='bottom', fontsize=9, color=TEAL,
              fontweight='bold')
    ax_B.text((rL + rR) / 2, R2_y - bar_ch / 2 - 1.4,
              "individual's latent belief /\nopinion pattern; randomly initialized",
              ha='center', va='top', fontsize=7.2, color='black')

    # ---- Row 3 (Period) ----
    box(ax_B, inp_cx, R3_y, 12, 7,
        'Survey year\n(1995)', fontsize=8.6)
    arrow(ax_B, inp_cx + 6, R3_y, bar_left - 0.5, R3_y)
    pL, pR, _ = vector_bar(ax_B, bar_left, R3_y, bar_n, PURPLE,
                            cell_w=bar_cw, cell_h=bar_ch, seed=33)
    ax_B.text(pL, R3_y + bar_ch / 2 + 1.4, 'Period embedding (50-d)',
              ha='left', va='bottom', fontsize=9, color=PURPLE,
              fontweight='bold')
    ax_B.text((pL + pR) / 2, R3_y - bar_ch / 2 - 1.4,
              'temporal / historical context;\nrandomly initialized',
              ha='center', va='top', fontsize=7.2, color='black')

    # ---- Curly brace gathering the three bars on the right ----
    brace_x = bar_right + 3.5
    brace_y_top = R1_y + bar_ch / 2 + 0.5
    brace_y_bot = R3_y - bar_ch / 2 - 0.5
    tip_x, tip_y = right_brace(ax_B, brace_x, brace_y_top, brace_y_bot,
                                depth=1.8, lw=1.4)

    # ---- Cross + Dense + output (all aligned at brace tip y) ----
    cl_cx = 77
    dl_cx = 87
    out_x = 94
    # Label "Concatenated embedding (150-d)" floats ABOVE the arrow between
    # the brace tip and Cross layers, in the empty horizontal corridor.
    arrow_left  = tip_x + 0.4
    arrow_right = cl_cx - 4.5
    ax_B.text((arrow_left + arrow_right) / 2, tip_y + 4.5,
              'Concatenated', ha='center', va='center', fontsize=8.4,
              fontweight='bold')
    ax_B.text((arrow_left + arrow_right) / 2, tip_y + 2.0,
              'embedding (150-d)', ha='center', va='center', fontsize=8.0,
              fontweight='bold')
    arrow(ax_B, arrow_left, tip_y, arrow_right, tip_y)
    box(ax_B, cl_cx, tip_y, 9, 14,
        'Cross\nlayers\n(× 3)', fontsize=9, fontweight='bold')
    ax_B.text(cl_cx, tip_y - 9.0,
              'higher-order\ninteractions',
              ha='center', va='top', fontsize=6.8, fontstyle='italic')

    arrow(ax_B, cl_cx + 4.5, tip_y, dl_cx - 4.5, tip_y)
    box(ax_B, dl_cx, tip_y, 9, 14,
        'Dense\nlayers\n(× 3)', fontsize=9, fontweight='bold')
    ax_B.text(dl_cx, tip_y - 9.0,
              'combine features &\npredict response',
              ha='center', va='top', fontsize=6.8, fontstyle='italic')

    arrow(ax_B, dl_cx + 4.5, tip_y, out_x - 2.0, tip_y)

    # P(Agree) horizontal compact label
    ax_B.text(out_x, tip_y + 1.5, 'P(Agree) ∈ [0, 1]',
              ha='center', va='bottom', fontsize=9, fontweight='bold')
    person(ax_B, out_x - 2, tip_y - 6, RED, scale=0.9,
           label='Agree', lab_dx=1.8, label_fontsize=8.5)
    person(ax_B, out_x - 2, tip_y - 13, BLUE, scale=0.9,
           label='Disagree', lab_dx=1.8, label_fontsize=8.5)

    # Legend (lower-right corner, two entries)
    leg_x, leg_y = 74, 14
    padlock(ax_B, leg_x, leg_y, scale=1.0)
    ax_B.text(leg_x + 1.6, leg_y, '= frozen pre-trained weights',
              ha='left', va='center', fontsize=7.8)
    leg_y2 = leg_y - 4
    for k in range(8):
        ax_B.add_patch(Rectangle((leg_x - 0.6 + k * 0.55, leg_y2 - 0.5),
                                 0.5, 1.0, fc=ORANGE,
                                 alpha=0.4 + 0.07 * k, ec=EDGE, lw=0.2,
                                 zorder=3))
    ax_B.text(leg_x + 4.4, leg_y2, '= embedding vector',
              ha='left', va='center', fontsize=7.8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / 'figure_conceptual.pdf'
    png = OUT_DIR / 'figure_conceptual.png'
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {pdf}')
    print(f'saved: {png}')


if __name__ == '__main__':
    main()
