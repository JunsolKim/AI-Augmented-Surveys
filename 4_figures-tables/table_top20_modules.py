import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_PARQUET = BASE / "data" / "fig_table_gen" / "top20_modules.parquet"
OUT_TEX = BASE / "output" / "table_top20_modules.tex"


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


def _rep_vars_cell(rep_vars: list[dict]) -> str:
    if not rep_vars:
        return ""
    lines = []
    for i, rv in enumerate(rep_vars):
        var = tex_escape(rv.get("variable", ""))
        label = tex_escape(rv.get("label", ""))
        entry = f"{var} ({label})" if label else var
        if i < len(rep_vars) - 1:
            entry = entry + r", \\"
        lines.append(entry)
    body = " ".join(lines)
    # Nested tabular uses a wrapping p-column (not `l`) so that long
    # variable_label text wraps inside the cell instead of overflowing into
    # the adjacent `# Variables` column.
    return r"\begin{tabular}[t]{@{}p{\linewidth}@{}}" + body + r"\end{tabular}"


def render_tex(df: pd.DataFrame) -> str:
    lines: list[str] = [
        r"\begin{longtable}{p{0.25\linewidth}p{0.5\linewidth}c}",
        r"\caption{Top 20 Modules by Number of Variables in the Module with "
        r"Representative Variables}\label{tab:top20_modules_updated}\\",
        r"\hline",
        r"\textbf{Module} & \textbf{Representative Variables} & \textbf{\# Variables} \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"\textbf{Module} & \textbf{Representative Variables} & \textbf{\# Variables} \\",
        r"\hline",
        r"\endhead",
        r"\hline \multicolumn{3}{r}{{Continued on next page}} \\",
        r"\endfoot",
        r"\hline",
        r"\endlastfoot",
    ]
    for _, r in df.iterrows():
        module = tex_escape(str(r["module"]))
        rep_cell = _rep_vars_cell(json.loads(r["rep_vars_json"]))
        n = int(r["n_variables"])
        lines.append(f"{module} & {rep_cell} & {n} \\\\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.read_parquet(IN_PARQUET)
    tex = render_tex(df)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(f"saved: {OUT_TEX} ({len(df)} rows)")


if __name__ == "__main__":
    main()
