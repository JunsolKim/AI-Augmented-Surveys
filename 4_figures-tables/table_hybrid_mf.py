from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_PQ = BASE / "data" / "fig_table_gen" / "concat_mf.parquet"
OUT_TEX = BASE / "output" / "table_hybrid_mf.tex"

MODEL_ORDER = [
    ("alpaca",          "Alpaca-7b (ours)"),
    ("mf",              "Matrix factorization"),
    ("concat_mf_tfidf", "MF + TF-IDF"),
    ("concat_mf_sbert", "MF + embedding (SBERT)"),
]
SCENARIO_ORDER = [
    ("impute",  "Missing data imputation"),
    ("partial", "Retrodiction"),
]


def fmt(v):
    if v is None or pd.isna(v):
        return "--"
    return f"{float(v):.3f}"


def main() -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(IN_PQ)
    agg = (df.groupby(["scenario", "model"])[["auc", "acc"]]
             .mean().reset_index())
    idx = {(r["scenario"], r["model"]): r for _, r in agg.iterrows()}

    lines = [
        r"\begin{table}[H]",
        r"\caption{\textbf{Table XX.} Text-aware matrix-factorization baselines.}",
        r"\label{tab:hybrid_mf}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Scenario & Model & AUC & Accuracy \\",
        r"\midrule",
    ]
    for sc, slabel in SCENARIO_ORDER:
        for i, (mkey, mlabel) in enumerate(MODEL_ORDER):
            r = idx.get((sc, mkey))
            if r is None:
                continue
            sc_cell = slabel if i == 0 else ""
            auc = fmt(r["auc"])
            acc = fmt(r["acc"])
            if mkey == "alpaca":
                lines.append(
                    f"{sc_cell} & \\textbf{{{mlabel}}} & "
                    f"\\textbf{{{auc}}} & \\textbf{{{acc}}} \\\\"
                )
            else:
                lines.append(f"{sc_cell} & {mlabel} & {auc} & {acc} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\medskip",
        (r"\parbox{\linewidth}{\footnotesize \textit{Notes.} \emph{MF + TF-IDF} augments the "
         r"user--item rating matrix with raw TF-IDF question-text features "
         r"and factorizes jointly; \emph{MF + embedding} "
         r"uses SBERT sentence embeddings in the same place. This follows "
         r"the hybrid / collective-factorization design of Singh and "
         r"Gordon (2008) and the feature-augmented variants discussed in "
         r"Koren et al.\ (2009). Metrics are averaged over the same three "
         r"held-out folds for each scenario; Alpaca is the proposed approach.}"),
        r"\end{table}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"saved: {OUT_TEX}")


if __name__ == "__main__":
    main()
