from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "data" / "fig_table_gen"
OUT = BASE / "output"

TARGET_VARS = ["marhomo1", "busing", "concong", "cohabit"]
SECTION_TITLES = {
    "marhomo1": "homosexual couples have the right to marry",
    "busing":   "favor busing of Black and white schoolchildren",
    "concong":  "confidence in U.S. Congress",
    "cohabit":  "lived with spouse before marriage",
}
SECTION_LETTERS = {"marhomo1": "A", "busing": "B", "concong": "C", "cohabit": "D"}


def _tex_escape(s):
    if s is None or pd.isna(s):
        return ""
    s = str(s)
    return (s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
             .replace("_", r"\_").replace("$", r"\$"))


def _truncate(s, n):
    if s is None or pd.isna(s):
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    out_csv = pd.read_parquet(GEN / "counterfactual_roper_surveys.parquet")
    csv_path = OUT / "table_counterfactual_roper_surveys.csv"
    out_csv.to_csv(csv_path, index=False)
    print(f"saved: {csv_path}  ({len(out_csv)} rows)")

    # ---- TeX ----
    lines = [
        r"\clearpage",
        r"\begin{longtable}{rlp{7.0cm}p{4.0cm}}",
        r"\caption{\textbf{Table XX.} Roper Center surveys used as external-validation ground truth for the counterfactual-trend example variables (panels A--D of Figure~6).}",
        r"\label{tab:counterfactual_roper_surveys} \\",
        r"\toprule",
        r"Year & GSS variable & Roper study title & Source \\",
        r"\midrule",
        r"\endfirsthead",
        r"",
        r"\multicolumn{4}{l}{\textit{(continued from previous page)}} \\",
        r"\toprule",
        r"Year & GSS variable & Roper study title & Source \\",
        r"\midrule",
        r"\endhead",
        r"",
        r"\midrule",
        r"\multicolumn{4}{r}{\textit{(continued on next page)}} \\",
        r"\endfoot",
        r"",
        r"\bottomrule",
        (r"\multicolumn{4}{p{15cm}}{\footnotesize \textit{Notes.} Matches "
         r"are filtered to US national-adult samples, LLM-verified "
         r"same-construct pairs with confidence $\geq 0.85$, simple binary "
         r"response mapping, and years in which the GSS did not field the "
         r"corresponding question. ``Source'' is the surveying / sponsoring "
         r"organization listed in Roper's study metadata.} \\"),
        r"\endlastfoot",
        r"",
    ]
    first_block = True
    for var in TARGET_VARS:
        sub = out_csv[out_csv["gss_variable"] == var].sort_values(
            ["year", "studyTitle"]).reset_index(drop=True)
        title = f"{SECTION_LETTERS[var]}. {SECTION_TITLES[var]}"
        if not first_block:
            lines.append(r"\midrule")
        first_block = False
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{title}}}}} \\")
        if len(sub) == 0:
            lines.append(r"\multicolumn{4}{l}{\textit{(no qualifying Roper polls)}} \\")
            continue
        for _, r in sub.iterrows():
            year = int(r["year"])
            studyTitle = _tex_escape(_truncate(r["studyTitle"], 90))
            surveyBy = _tex_escape(_truncate(r["surveyBy"], 50))
            lines.append(
                rf"{year} & {var} & {studyTitle} & {surveyBy} \\"
            )

    lines.extend([
        r"\end{longtable}",
    ])
    tex_path = OUT / "table_counterfactual_roper_surveys.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"saved: {tex_path}")


if __name__ == "__main__":
    main()
