from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_CSV = BASE / "data" / "fig_table_gen" / "accuracythresholds_alpaca.csv"
OUT_TEX = BASE / "output" / "table_accuracy_thresholds.tex"

SCENARIO_ORDER = ["impute", "partial", "total"]


def render_tex(df: pd.DataFrame) -> str:
    pivot = (df.pivot_table(index="threshold_pct",
                            columns="scenario",
                            values="pct_within")
               .reindex(columns=SCENARIO_ORDER)
               .sort_index())

    body_lines = []
    for t_pct, row in pivot.iterrows():
        cells = [f"{row[s]:.1f}" for s in SCENARIO_ORDER]
        body_lines.append(
            f"        within {int(t_pct)}\\% & " + " & ".join(cells) + r" \\"
        )
    body = "\n".join(body_lines)

    tex = (
r"""\begin{table}[ht]
    \centering
    \caption{Percentage of opinions predicted by different models within varying accuracy thresholds}
    \label{tab:model_prediction_accuracy}
    \begin{tabular}{lccc}
        \hline
        \textbf{Prediction intervals} & \textbf{Imputation (\%)} & \textbf{Retrodiction (\%)} & \textbf{Unasked prediction (\%)} \\
        \hline
""" + body + r"""
        \hline
    \end{tabular}
\end{table}
"""
    )
    return tex


def main() -> None:
    df = pd.read_csv(IN_CSV)
    tex = render_tex(df)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(f"saved: {OUT_TEX}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
