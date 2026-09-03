from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


BASE = Path(__file__).resolve().parents[1]
IN_PATH = BASE / "data" / "fig_table_gen" / "samesexframing_paired_individual.parquet"
META_PATH = BASE / "data" / "gss_train_vars_corrected_binarized_again.parquet"
OUT_TEX = BASE / "output" / "table_samesex_framing.tex"


def _fmt_p(p: float) -> str:
    return "$< .001$" if p < 1e-3 else f"${p:.3f}$".replace("0.", ".")


def _wilcoxon_z(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Return (z, p, n_nonzero). z is the normal-approximation statistic for
    the Wilcoxon signed-rank test (tie- and zero-corrected)."""
    d = x - y
    d = d[d != 0]
    n = len(d)
    ranks = rankdata(np.abs(d))
    t_plus = float(ranks[d > 0].sum())
    mean_t = n * (n + 1) / 4.0
    # Tie correction to the variance of T+.
    _, counts = np.unique(ranks, return_counts=True)
    tie_term = float(((counts**3 - counts).sum()) / 48.0)
    var_t = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term
    z = (t_plus - mean_t) / np.sqrt(var_t)
    # Two-sided p-value via scipy's wilcoxon for consistency.
    _, p = wilcoxon(x, y, method="approx")
    return z, float(p), n


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    m1 = df["pred_marhomo1_cal"].to_numpy(dtype=np.float64)
    m2 = df["pred_marhomo_cal"].to_numpy(dtype=np.float64)

    mean_m1 = float(m1.mean()) * 100
    mean_m2 = float(m2.mean()) * 100
    diff_pp = float((m2 - m1).mean()) * 100

    z, p, n_nz = _wilcoxon_z(m2, m1)

    meta = pd.read_parquet(
        META_PATH, columns=["variable_name", "processed_question"]
    ).set_index("variable_name")
    q1 = meta.loc["marhomo1", "processed_question"]
    q2 = meta.loc["marhomo", "processed_question"]
    # Italicize the distinguishing word "should" in the marhomo question.
    q2_emph = q2.replace("should", r"\emph{should}")

    year_min, year_max = int(df["year"].min()), int(df["year"].max())

    diff_row = (
        r"\textit{Paired difference}, Wilcoxon signed-rank test: "
        f"$z = {z:.2f}$, $p$ {_fmt_p(p)}, $N = {len(df):,}$"
        f" & ${diff_pp:+.2f}$ \\\\"
    )

    caption = (
        r"Effect of the word ``should'' on predicted agreement with same-sex "
        r"marriage rights."
    )

    note = (
        r"\emph{Note.} Each respondent contributes a paired prediction for "
        r"both items."
    )

    lines = [
        r"\begin{table}[htp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{tab:samesex_framing}",
        r"\begin{tabular}{p{9cm} r}",
        r"\toprule",
        r"Survey question & Mean (\%) \\",
        r"\midrule",
        f"``{q1}'' & {mean_m1:.2f} \\\\",
        r"\addlinespace",
        f"``{q2_emph}'' & {mean_m2:.2f} \\\\",
        r"\midrule",
        diff_row,
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{0.5em}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        note,
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    OUT_TEX.parent.mkdir(exist_ok=True)
    OUT_TEX.write_text("\n".join(lines))

    print(f"saved: {OUT_TEX}")
    print(f"  N = {len(df):,}  ({year_min}–{year_max})")
    print(f"  marhomo1 mean = {mean_m1:.2f}%")
    print(f"  marhomo  mean = {mean_m2:.2f}%")
    print(f"  Δ = {diff_pp:+.2f} pp")
    print(f"  Wilcoxon: z = {z:.2f}, p = {p:.2e}, n_nonzero = {n_nz:,}")


if __name__ == "__main__":
    main()
