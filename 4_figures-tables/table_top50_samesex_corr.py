from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_PARQUET = BASE / "data" / "fig_table_gen" / "top50_samesex_corr.parquet"
OUT_TEX = BASE / "output" / "table_samesex_50.tex"

NOTE = (
    "Note. This table highlights the top 50 variables ranked by their "
    "absolute correlation with the target question about opinions on "
    "same-sex marriage."
)


_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "_": r"\_",
    "#": r"\#",
    "%": r"\%",
    "&": r"\&",
    "$": r"\$",
    "^": r"\^{}",
    "~": r"\~{}",
}


def tex_escape(s: str) -> str:
    if not isinstance(s, str):
        return ""
    for k, v in _TEX_ESCAPES.items():
        s = s.replace(k, v)
    return s


def render_tex(df: pd.DataFrame) -> str:
    lines: list[str] = [
        r"\begin{longtable}{clp{7cm}c}",
        r"\caption{List of the 50 Survey Questions Most Closely Correlated "
        r"with Same-Sex Marriage Opinions.} \\ \hline",
        r"",
        r"\textbf{Rank} & \textbf{Variable} & \textbf{Description} & "
        r"\textbf{Abs(Correlation)} \\ \hline",
    ]
    for _, r in df.iterrows():
        rank = int(r["rank"])
        var = tex_escape(str(r["variable"]))
        desc = tex_escape(str(r["description"]))
        corr = f"{float(r['abs_corr']):.3f}"
        lines.append(f"{rank} & {var} & {desc} & {corr} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{longtable}")
    lines.append(r"\noindent")
    lines.append(NOTE)
    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.read_parquet(IN_PARQUET)
    tex = render_tex(df)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(f"saved: {OUT_TEX} ({len(df)} rows)")


if __name__ == "__main__":
    main()
