from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
METRICS_CSV = BASE / "data" / "fig_table_gen" / "retro_interp_forecast_metrics.csv"
OUT_TEX = BASE / "output" / "table_retro_interp_forecast.tex"

MODE_ORDER = ["retrodiction", "interpolation", "forecast"]
MODE_LABELS = {
    "retrodiction":  "Backcasting",
    "interpolation": "Interpolation",
    "forecast":      "Forecasting",
}


def fmt(v):
    if v is None or pd.isna(v):
        return "--"
    return f"{float(v):.3f}"


def main() -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS_CSV)
    row_by_mode = {r["mode"]: r for _, r in df.iterrows()}

    lines = [
        r"\begin{table}[H]",
        r"\caption{\textbf{Table XX.} Alpaca individual-level performance, by retrodiction scenario.}",
        r"\label{tab:retro_interp_forecast}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scenario & $N$ & AUC & Accuracy & F1 \\",
        r"\midrule",
    ]
    for mode in MODE_ORDER:
        r = row_by_mode.get(mode)
        if r is None:
            lines.append(f"{MODE_LABELS[mode]} & -- & -- & -- & -- \\\\")
            continue
        n = "--" if pd.isna(r["n"]) else f"{int(r['n']):,}"
        lines.append(
            f"{MODE_LABELS[mode]} & {n} & "
            f"{fmt(r['auc'])} & {fmt(r['acc'])} & {fmt(r['f1'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\medskip",
        (r"\parbox{\linewidth}{\footnotesize \textit{Notes.} Each held-out "
         r"validation cell $(v, y)$ is classified by where $y$ sits relative "
         r"to the variable's other observation years: backcasting if $y$ is "
         r"before the earliest, forecasting if $y$ is after the latest, "
         r"otherwise interpolation. $N$ is the number of held-out (respondent, "
         r"variable, year) cells pooled across the partial-CV folds.}"),
        r"\end{table}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"saved: {OUT_TEX}")


if __name__ == "__main__":
    main()
