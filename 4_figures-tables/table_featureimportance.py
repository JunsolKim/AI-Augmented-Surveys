from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_PATH = BASE / "data" / "fig_table_gen" / "featureimportance_permutation.parquet"
OUT_TEX = BASE / "output" / "table_featureimportance.tex"

# Column order.
COLUMN_ORDER = ["Original", "Shuffle Q", "Shuffle R", "Shuffle P"]
METRIC_ORDER = [("auc", "AUC"), ("accuracy", "Accuracy"), ("f1", "F1-score")]


def main() -> None:
    df = pd.read_parquet(IN_PATH).set_index("condition").loc[COLUMN_ORDER]

    body_lines = []
    for col, label in METRIC_ORDER:
        vals = " & ".join(f"{df.loc[c, col]:.3f}" for c in COLUMN_ORDER)
        body_lines.append(f"{label:<10} & {vals} \\\\")
    body = "\n".join(body_lines)

    tex = r"""\begin{table}[ht]
\caption{\textbf{Table XX.} Feature importance from permutation experiments.}
\label{tab:metrics}
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Metric} &
\shortstack{\textbf{Retrodiction}\\\textbf{(Original)}} &
\shortstack{\textbf{Shuffling}\\\textbf{Question}\\\textbf{Embedding}} &
\shortstack{\textbf{Shuffling}\\\textbf{Respondent}\\\textbf{Embedding}} &
\shortstack{\textbf{Shuffling}\\\textbf{Period}\\\textbf{Embedding}} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}

\medskip
\parbox{\linewidth}{\footnotesize \textit{Notes.} We randomly shuffle one embedding at a time while keeping all others unchanged, and report the resulting drop in retrodiction performance (AUC, Accuracy, F1). The larger the drop, the more important the embedding.}
\end{table}
"""
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)

    for col, label in METRIC_ORDER:
        row = " ".join(f"{c}={df.loc[c, col]:.3f}" for c in COLUMN_ORDER)
        print(f"  {label:<10}  {row}")
    print(f"Saved: {OUT_TEX}")


if __name__ == "__main__":
    main()
