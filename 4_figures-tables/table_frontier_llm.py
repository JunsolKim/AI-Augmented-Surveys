from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
FRONTIER_CSV = BASE / "data" / "fig_table_gen" / "frontier_llm_metrics.csv"
MODELCOMP_CSV = BASE / "data" / "fig_table_gen" / "modelcomparison_metrics.csv"
OUT_TEX = BASE / "output" / "table_frontier_llm.tex"

LLM_MODELS = [
    ("gpt",              "GPT-4o"),
    ("gemini-2.5-flash", "Gemini-2.5-flash"),
]
LLM_STRATEGIES = [
    ("demographics",       "Demographics"),
    ("random_k10",         "Random-$k{=}10$"),
    ("correlation_topk10", "Correlation-top-$k{=}10$"),
]


def fmt(v):
    if v is None or pd.isna(v):
        return "--"
    return f"{float(v):.3f}"


def main() -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    fl = pd.read_csv(FRONTIER_CSV)
    mc = pd.read_csv(MODELCOMP_CSV)

    def fl_lookup(scenario, model, strategy):
        r = fl.loc[(fl["scenario"] == scenario)
                   & (fl["model"] == model)
                   & (fl["strategy"] == strategy)]
        if len(r) == 0:
            return None, None
        return float(r.iloc[0]["accuracy"]), float(r.iloc[0]["f1"])

    # Alpaca scenario rows -- all evaluated on the same GSS 2018 validation
    # intersection used for the frontier-LLM cells.
    alpaca_imp_acc, alpaca_imp_f1 = fl_lookup("impute", "alpaca", "all-intersection")
    alpaca_retro_acc, alpaca_retro_f1 = fl_lookup("partial", "alpaca", "all-intersection")
    alpaca_unasked_acc, alpaca_unasked_f1 = fl_lookup("total", "alpaca", "all-intersection")

    lines = [
        r"\begin{table}[H]",
        r"\caption{\textbf{Table XX.} Fine-tuned Alpaca-7b vs.\ prompted frontier LLMs on the GSS 2018 validation data.}",
        r"\label{tab:frontier_llm}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Model & Strategy & Accuracy & F1 \\",
        r"\midrule",
        r"\multirow{3}{*}{Alpaca-7b}",
        rf" & Imputation   & \textbf{{{fmt(alpaca_imp_acc)}}}  & \textbf{{{fmt(alpaca_imp_f1)}}}  \\",
        rf" & Retrodiction & \textbf{{{fmt(alpaca_retro_acc)}}} & \textbf{{{fmt(alpaca_retro_f1)}}} \\",
        rf" & Unasked      & \textbf{{{fmt(alpaca_unasked_acc)}}} & \textbf{{{fmt(alpaca_unasked_f1)}}} \\",
    ]
    for mkey, mlabel in LLM_MODELS:
        lines.append(r"\midrule")
        lines.append(rf"\multirow{{3}}{{*}}{{{mlabel}}}")
        for skey, slab in LLM_STRATEGIES:
            acc, f1 = fl_lookup("impute", mkey, skey)
            lines.append(rf" & {slab} & {fmt(acc)} & {fmt(f1)} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\medskip",
        (r"\parbox{\linewidth}{\footnotesize \textit{Notes.} Alpaca-7b "
         r"entries (in bold) are the fine-tuned baseline evaluated on the "
         r"three scenarios (imputation, retrodiction, unasked). Frontier-LLM "
         r"entries show three prompting strategies (Demographics, "
         r"Random-$k{=}10$ context, Correlation-top-$k{=}10$ context) "
         r"evaluated on the GSS 2018 data.}"),
        r"\end{table}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"saved: {OUT_TEX}")


if __name__ == "__main__":
    main()
