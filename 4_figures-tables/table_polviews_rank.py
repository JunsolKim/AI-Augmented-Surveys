from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_PQ = BASE / "data" / "fig_table_gen" / "polviews_rank.parquet"
OUT_TEX = BASE / "output" / "table_polviews_rank.tex"
OUT_CSV = BASE / "output" / "table_polviews_rank.csv"

N_TOP = 40


def _escape(s: str) -> str:
    s = str(s).replace("&", r"\&").replace("%", r"\%") \
              .replace("_", r"\_").replace("#", r"\#") \
              .replace("$", r"\$")
    return s


def _norm(s: str) -> str:
    """Whitespace-normalize and replace stray control / smart-quote chars
    (Windows-1252 ’ “ ” → ASCII)."""
    s = str(s).replace("\n", " ").replace("\r", " ").strip()
    s = (s.replace("", "'").replace("", "'")
          .replace("", '"').replace("", '"')
          .replace("‘", "'").replace("’", "'")
          .replace("“", '"').replace("”", '"'))
    return " ".join(s.split())


def _block(lines: list, df: pd.DataFrame, heading: str) -> None:
    lines.append(r"\multicolumn{5}{l}{\textit{" + heading + r"}} \\")
    for _, r in df.iterrows():
        var = _escape(str(r["variable"]))
        desc = _escape(_norm(str(r.get("description", "") or r.get("question", ""))))
        rho = f"{float(r['spearman_rho']):+.3f}"
        auc = r["alpaca_partial_auc"]
        auc_s = "--" if pd.isna(auc) else f"{float(auc):.3f}"
        lines.append(
            f"{int(r['rank'])} & {var} & {desc} & "
            f"{rho} & {auc_s} \\\\"
        )


def main() -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(IN_PQ)
    df = df.sort_values("abs_rho", ascending=False).reset_index(drop=True)

    top = df.head(N_TOP).copy()
    top["rank"] = range(1, len(top) + 1)
    top["group"] = "top"

    bot = df.tail(N_TOP).sort_values("abs_rho", ascending=True).copy()
    bot["rank"] = range(1, len(bot) + 1)
    bot["group"] = "bottom"

    both = pd.concat([top, bot], ignore_index=True)
    both.to_csv(OUT_CSV, index=False)
    print(f"saved: {OUT_CSV}")

    lines = [
        r"{\small",
        r"\begin{longtable}{rlp{8cm}rr}",
        r"\caption{\textbf{Table XX.} GSS variables ranked by absolute Spearman correlation with the 7-point political-ideology scale (polviews).}",
        r"\label{tab:polviews_rank} \\",
        r"\toprule",
        r"Rank & Variable & Description & $\rho$(polviews) & Alpaca AUC \\",
        r"\midrule",
        r"\endhead",
    ]
    _block(lines, top, f"Top-{N_TOP} most ideology-correlated")
    lines.append(r"\midrule")
    _block(lines, bot, f"Bottom-{N_TOP} least ideology-correlated")
    lines += [
        r"\bottomrule",
        (r"\multicolumn{5}{p{15cm}}{\footnotesize \textit{Notes.} $\rho$ is "
         r"the signed Spearman correlation with polviews (positive: "
         r"conservative respondents are more likely to endorse). "
         r"Alpaca AUC is the out-of-fold partial-task AUC per variable.} \\"),
        r"\end{longtable}",
        r"}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"saved: {OUT_TEX}")


if __name__ == "__main__":
    main()
