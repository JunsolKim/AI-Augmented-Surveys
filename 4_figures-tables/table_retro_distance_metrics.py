from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_CSV = BASE / "data" / "fig_table_gen" / "retro_distance_metrics.csv"
OUT_TEX = BASE / "output" / "table_retro_distance_metrics.tex"

BIN_ORDER = ["1", "2-3", "4-5", "6-7", "8+"]
MODE_ROWS = [
    ("retrodiction",  "A. Backcasting"),
    ("interpolation", "B. Interpolation"),
    ("forecast",      "C. Forecasting"),
]


def fmt(v, places=3):
    if v is None or pd.isna(v):
        return "--"
    return f"{float(v):.{places}f}"


def fmt_n(v):
    if v is None or pd.isna(v):
        return "--"
    return f"{int(v):,}"


def main() -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    idx = {(r["mode"], r["dist_bin"]): r for _, r in df.iterrows()}

    lines = [
        r"\begin{table}[H]",
        r"\caption{\textbf{Table XX.} Alpaca individual-level performance, by retrodiction scenario and distance to the nearest training year.}",
        r"\label{tab:retro_distance_metrics}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Scenario & Distance to nearest training year & $N$ & AUC & Accuracy & F1 \\",
        r"\midrule",
    ]
    for mode_key, mode_label in MODE_ROWS:
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{mode_label}}}}} \\")
        for b in BIN_ORDER:
            r = idx.get((mode_key, b))
            if r is None:
                cells = ["--"] * 4
            else:
                cells = [fmt_n(r["n"]), fmt(r["auc"]),
                         fmt(r["acc"]), fmt(r["f1"])]
            label_b = b if b != "8+" else r"$8+$"
            lines.append(f" & {label_b} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\medskip",
        (r"\parbox{\linewidth}{\footnotesize \textit{Notes.} Each held-out "
         r"validation cell is classified by scenario (\emph{backcasting}: "
         r"held-out year before the variable's earliest training year; "
         r"\emph{forecasting}: after the latest; \emph{interpolation}: in "
         r"between) and by its distance, in years, to the variable's nearest "
         r"training year in the same fold. AUC, Accuracy, and F1 are computed "
         r"over individual (respondent, variable, year) cells pooled across "
         r"the 10 partial-CV folds.}"),
        r"\end{table}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"saved: {OUT_TEX}")


if __name__ == "__main__":
    main()
