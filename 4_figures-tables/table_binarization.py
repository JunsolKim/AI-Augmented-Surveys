import json
import re
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PARQUET = REPO / "data" / "gss_train_vars_corrected_binarized_again.parquet"
OUT_TEX = REPO / "output" / "table_binary_transformation.tex"

TOP_N = 50

_PREFIX_RE = re.compile(r"^\d+:\s*")

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
    for k, v in _TEX_ESCAPES.items():
        s = s.replace(k, v)
    return s


def parse_binarized(binarized_json: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return (labels, values) with null-mapped entries dropped."""
    d = json.loads(binarized_json)
    labels, values = [], []
    for k, v in d.items():
        if v is None:
            continue
        label = _PREFIX_RE.sub("", k).strip().lower()
        labels.append(label)
        values.append(int(v))
    return tuple(labels), tuple(values)


def build_rows(df: pd.DataFrame, top_n: int) -> list[tuple[int, str, int, str]]:
    pairs = df["binarized"].apply(parse_binarized)
    df = df.assign(
        labels=pairs.apply(lambda x: x[0]),
        values=pairs.apply(lambda x: x[1]),
    )
    grouped = (
        df.groupby(["labels", "values"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    rows = []
    for i, r in enumerate(grouped.itertuples(index=False), start=1):
        rows.append((
            i,
            ", ".join(r.labels),
            int(r.n),
            ", ".join(str(v) for v in r.values),
        ))
    return rows


def render_tex(rows: list[tuple[int, str, int, str]]) -> str:
    header = (
        r"\textbf{Rank} & \textbf{Response Options} "
        r"& \textbf{N} & \textbf{Binarized} \\"
    )
    lines = [
        r"\begin{longtable}{r p{9cm} r p{3cm}}",
        r"\caption{Binary Transformation of Survey Response Options: Top 50 responses.} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{4}{c}{\textit{(continued from previous page)}} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{4}{r}{\textit{(continued on next page)}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for rank, resp, n, bin_ in rows:
        lines.append(
            f"{rank} & {tex_escape(resp)} & {n} & {tex_escape(bin_)} \\\\"
        )
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df = df[df["use_variable_final"] == 1].copy()

    rows = build_rows(df, TOP_N)
    tex = render_tex(rows)

    OUT_TEX.parent.mkdir(exist_ok=True)
    OUT_TEX.write_text(tex)

    print(f"saved: {OUT_TEX} ({len(rows)} rows)")
    print(f"top 10 patterns:")
    for rank, resp, n, bin_ in rows[:10]:
        resp_short = resp if len(resp) < 70 else resp[:67] + "..."
        print(f"  {rank:>2}  n={n:>4}  [{bin_}]  {resp_short}")


if __name__ == "__main__":
    main()
