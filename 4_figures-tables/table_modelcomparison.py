from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
METRICS_CSV = BASE / "data" / "fig_table_gen" / "modelcomparison_metrics.csv"
OUT_TEX = BASE / "output" / "table_modelcomparison.tex"

MODEL_ORDER = ["alpaca", "gptj", "roberta", "mf", "mice"]
MODEL_HEADERS = {
    "alpaca":  "Alpaca-7b",
    "gptj":    "GPT-J-6b",
    "roberta": "RoBERTa-large",
    "mf":      r"\shortstack{Matrix\\Factorization}",
    "mice":    "MICE",
}

SCENARIO_COL_WIDTH = "2.3cm"
METRIC_COL_WIDTH   = "1.6cm"
MODEL_COL_WIDTH    = "1.9cm"

SCENARIO_ORDER = ["impute", "partial", "total"]
SCENARIO_LABELS = {
    "impute":  "Missing data imputation",
    "partial": "Retrodiction",
    "total":   "Unasked opinion prediction",
}

METRICS = [("auc", "AUC"), ("acc", "Accuracy"), ("f1", "F1-score")]


def fmt_cell(val: float | None, is_best: bool) -> str:
    if val is None or pd.isna(val):
        return ""
    s = f"{val:.3f}"
    return f"\\textbf{{{s}}}" if is_best else s


def build_body(df: pd.DataFrame) -> list[str]:
    lookup: dict[tuple[str, str, str], float | None] = {}
    for _, r in df.iterrows():
        for mk, _ in METRICS:
            v = r[mk]
            lookup[(r["model"], r["scenario"], mk)] = (
                None if pd.isna(v) else float(v)
            )

    lines: list[str] = []
    for si, skey in enumerate(SCENARIO_ORDER):
        for mi, (mk, mlabel) in enumerate(METRICS):
            vals = [lookup.get((mdl, skey, mk)) for mdl in MODEL_ORDER]
            numeric = [v for v in vals if v is not None]
            best = max(numeric) if numeric else None
            cells = [
                fmt_cell(v, is_best=(v is not None and best is not None
                                      and v == best))
                for v in vals
            ]

            if mi == 0:
                row_head = (
                    f"\\multirow{{3}}{{{SCENARIO_COL_WIDTH}}}{{"
                    f"{SCENARIO_LABELS[skey]}}} "
                )
            else:
                row_head = " "
            row = f"    {row_head}& {mlabel} & " + " & ".join(cells) + r" \\"
            if mi == 2 and si < len(SCENARIO_ORDER) - 1:
                row += r" \midrule"
            elif mi == 2:
                row += r" \bottomrule"
            lines.append(row)

    return lines


def render_tex(df: pd.DataFrame) -> str:
    headers = " & ".join(MODEL_HEADERS[m] for m in MODEL_ORDER)
    n_model = len(MODEL_ORDER)
    n_col = n_model + 2
    cmid_end = n_col
    # Wider label columns so scenario text ("Unasked opinion prediction")
    # wraps within the multirow cell instead of spilling into neighbors, and
    # wider model columns so "Matrix Factorization" and "RoBERTa-large" do
    # not get hyphenated across lines.
    col_spec = (
        f"@{{}}p{{{SCENARIO_COL_WIDTH}}} p{{{METRIC_COL_WIDTH}}} "
        + " ".join(
            [r">{\centering\arraybackslash}p{" + MODEL_COL_WIDTH + "}"]
            * n_model
        )
        + "@{}"
    )

    lines = [
        r"\begin{table}[htbp]",
        r"  \caption{\textbf{Table XX.} Prediction performance across five models and three scenarios.}",
        r"  \label{tab:modelcomparison}",
        r"  \centering",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        (r"    & & "
         f"\\multicolumn{{{n_model}}}{{c}}{{Models}} \\\\ "
         f"\\cmidrule(l){{3-{cmid_end}}}"),
        f"     &  & {headers} \\\\ \\midrule",
    ]
    lines.extend(build_body(df))
    lines.append(r"  \end{tabular}")
    lines.append(r"")
    lines.append(r"  \medskip")
    note = (
        r"  \parbox{\linewidth}{\footnotesize \textit{Notes.} The "
        r"best-performing models are highlighted in bold. AUC measures the "
        r"probability of ranking a randomly selected positive response above "
        r"a randomly selected negative response. Accuracy is (TP + TN) / "
        r"(TP + FP + TN + FN), and F1 is $2 \cdot (\mathrm{precision} \cdot "
        r"\mathrm{recall}) / (\mathrm{precision} + \mathrm{recall})$. Matrix "
        r"factorization and MICE cannot be applied to unasked-opinion "
        r"prediction.}"
    )
    lines.append(note)
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.read_csv(METRICS_CSV)
    tex = render_tex(df)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(f"saved: {OUT_TEX}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
