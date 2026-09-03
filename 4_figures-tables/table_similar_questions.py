from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_PATH = BASE / "data" / "fig_table_gen" / "similar_questions.parquet"
OUT_TEX = BASE / "output" / "table_similar_questions.tex"

TARGET_VARS = ["homosex", "marhomo1", "busing", "nomeat"]
SPLIT_AFTER = "marhomo1"

HEADER_ROW = (
    r" \toprule  " + "\n"
    + r"\textbf{Survey questions} & & \textbf{Most similar survey questions} & & "
    + r"\textbf{Cosine \linebreak similarity} \\ \midrule" + "\n"
)

NOTE = (
    "Note: For each survey question in Figure 4, we present the most similar "
    "survey questions in the question embedding space. The variable names are "
    "presented in parenthesis. The similarity between two survey questions is "
    "measured by the cosine similarity between the two question embeddings of "
    "the two survey questions."
)


def _tex_escape(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
             .replace("#", r"\#").replace("_", r"\_").replace("{", r"\{")
             .replace("}", r"\}").replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}"))


def _block(df: pd.DataFrame, targets: list[str]) -> str:
    """Build the body of one longtable (rows for a subset of targets)."""
    out = []
    for i, tv in enumerate(targets):
        sub = df[df["target"] == tv].sort_values("rank")
        if sub.empty:
            continue
        n = len(sub)
        first = sub.iloc[0]
        target_cell = rf"\multirow{{{n}}}{{5cm}}{{({_tex_escape(tv)}) " \
                      rf"{_tex_escape(first['target_question'])}}}"
        sv = _tex_escape(first["similar_variable"])
        sq = _tex_escape(first["similar_question"])
        out.append(f"{target_cell} & & ({sv}) {sq} & & {first['cosine']:.3f} \\\\")
        for _, r in sub.iloc[1:].iterrows():
            sv = _tex_escape(r["similar_variable"])
            sq = _tex_escape(r["similar_question"])
            out.append(f" & & ({sv}) {sq} & & {r['cosine']:.3f} \\\\")
        if i < len(targets) - 1:
            out.append(r" \hline")
    return "\n".join(out)


def _longtable(body: str, caption: str, label: str | None = None) -> str:
    colspec = (
        r"p{5cm}@{}" + "\n"
        + r"p{0.3cm}@{}" + "\n"
        + r">{\raggedright}p{8cm}@{}" + "\n"
        + r"p{0.3cm}@{}" + "\n"
        + r"p{2cm}@{}" + "\n"
    )
    firsthead = "\\\\\n" + HEADER_ROW + "\\endfirsthead\n"
    head = "\\\\\n" + HEADER_ROW + "\\endhead\n"
    label_line = f"\\label{{{label}}}\n" if label else ""
    return (
        "\\begin{longtable}{\n" + colspec + "}\n"
        + f"\\caption{{\\textbf{{{caption}}} }}\n"
        + firsthead
        + head
        + "\n"
        + body + "\n"
        + "\\bottomrule\n"
        + label_line
        + "\\end{longtable}\n"
    )


def main():
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(IN_PATH)

    split_idx = TARGET_VARS.index(SPLIT_AFTER) + 1
    first_targets = TARGET_VARS[:split_idx]
    second_targets = TARGET_VARS[split_idx:]

    part1 = _longtable(
        _block(df, first_targets),
        caption="Most similar survey questions.",
        label="table:tables2",
    )
    part2 = _longtable(
        _block(df, second_targets),
        caption="Most similar survey questions (continued).",
        label=None,
    )

    tex = (
        part1
        + "\n\\clearpage\n\\setcounter{table}{3}\n"
        + part2
        + f"\n\\noindent {NOTE}\n"
    )

    OUT_TEX.write_text(tex)
    print(f"Saved: {OUT_TEX} ({len(df)} rows; split after '{SPLIT_AFTER}')")


if __name__ == "__main__":
    main()
