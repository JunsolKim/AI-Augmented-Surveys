from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_CSV = BASE / "data" / "fig_table_gen" / "perfbygroup_metrics.csv"
OUT_TEX = BASE / "output" / "table_perfbygroup.tex"

METHODS = [
    ("our",          "Our Method"),
    ("reg",          "Regression"),
    ("reg_logit",    "Logistic"),
    ("reg_sq",       r"Reg+Sq"),
    ("reg_sq_logit", "Logistic+Sq"),
    ("mf",           "MF"),
]

# Group blocks: (section title or None, [(scenario, group, label), ...])
BLOCKS: list[tuple[str | None, list[tuple[str, str, str]]]] = [
    (None,                            [("All",               "All",     "All")]),
    ("A. Sparsity",                   [("Sparsity",          "year=1",  "year = 1"),
                                       ("Sparsity",          "year=2",  "year = 2"),
                                       ("Sparsity",          "year>2",  "year $>$ 2")]),
    ("B. Volatility Level",           [("Volatility",        "high",    "High volatility"),
                                       ("Volatility",        "low",     "Low volatility")]),
    ("C. Temporal Distance to the Nearest Year",
                                      [("Temporal distance", "dist>=3", r"Distance $\geq$ 3"),
                                       ("Temporal distance", "dist<3",  r"Distance $<$ 3")]),
]


def _fmt(v: float | None) -> str:
    if v is None or pd.isna(v):
        return ""
    s = f"{v:.3f}"
    # drop the leading 0 so cells read like ".985"
    return s[1:] if s.startswith("0.") else s


def _row_cells(row: pd.Series) -> list[str]:
    """One combined ``ρ/MAE`` cell per method. Bold the cell with the
    highest ρ in this row."""
    out: list[str] = []
    rs_valid = [(k, row.get(f"{k}_r")) for k, _ in METHODS
                if pd.notna(row.get(f"{k}_r"))]
    best_key = max(rs_valid, key=lambda kv: kv[1])[0] if rs_valid else None
    for k, _ in METHODS:
        rv = row.get(f"{k}_r")
        mv = row.get(f"{k}_mae")
        if pd.isna(rv) and pd.isna(mv):
            cell = ""
        else:
            cell = f"{_fmt(rv)}/{_fmt(mv)}"
        if k == best_key and cell:
            cell = f"\\textbf{{{cell}}}"
        out.append(cell)
    return out


def render_tex(df: pd.DataFrame) -> str:
    lookup = {(r["scenario"], r["group"]): r for _, r in df.iterrows()}

    n_methods = len(METHODS)
    col_spec = "l" + "rr" + "c" * n_methods
    n_total_cols = 1 + 2 + n_methods

    # Header: method names with combined "(ρ / MAE)" sub-label
    method_head = " & ".join(f"\\textbf{{{lab}}}" for _, lab in METHODS)
    sub_head = " & ".join([r"($\rho$ / MAE)"] * n_methods)

    lines: list[str] = [
        r"\clearpage",
        r"\begin{sidewaystable}",
        r"  \caption{\textbf{Table XX.} Performance by sparsity, temporal distance, and volatility.}",
        r"  \label{tab:perfbygroup}",
        r"  \centering",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        (f"    \\textbf{{Scenario}} & $N_{{\\text{{var, year}}}}$ "
         f"& $N_{{\\text{{var}}}}$ & {method_head} \\\\"),
        f"     &  &  & {sub_head} \\\\",
        r"    \midrule",
    ]

    for bi, (section, items) in enumerate(BLOCKS):
        if section is not None:
            lines.append(
                f"    \\multicolumn{{{n_total_cols}}}{{l}}"
                f"{{\\textit{{{section}}}}} \\\\"
            )
        for scen, grp, lab in items:
            row = lookup.get((scen, grp))
            if row is None:
                cells = ["N/A"] * (2 + 2 * n_methods)
            else:
                cells = [str(int(row["n_var_year"])),
                         str(int(row["n_var"]))] + _row_cells(row)
            lines.append(f"    {lab} & " + " & ".join(cells) + r" \\")
        if bi < len(BLOCKS) - 1:
            lines.append(r"    \midrule")
        else:
            lines.append(r"    \bottomrule")

    lines.append(r"  \end{tabular}")
    lines.append(r"")
    lines.append(r"  \medskip")
    note = (
        r"  \parbox{\linewidth}{\footnotesize \textit{Notes.} Scenario "
        r"describes the condition under which survey questions are missing. "
        r"$N_{\text{var, year}}$ is the number of variable--year entries; "
        r"$N_{\text{var}}$ is the number of distinct GSS variables in each "
        r"group. Each cell reports $\rho$ / MAE, where $\rho$ is the "
        r"Spearman correlation between predicted and observed population "
        r"averages and MAE is the mean absolute error. ``Regression'' is "
        r"linear time-series regression (OLS, $\text{obs\_wavg} \sim "
        r"\text{year}$); ``Logistic'' is the same regression with a "
        r"Binomial logit link; ``Reg+Sq'' adds a quadratic year term; "
        r"``Logistic+Sq'' is the quadratic regression with a Binomial "
        r"logit link; ``MF'' is the matrix-factorization baseline. "
        r"\textbf{Bold} indicates the highest~$\rho$ in each row.}"
    )
    lines.append(note)
    lines.append(r"\end{sidewaystable}")
    lines.append(r"\clearpage")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.read_csv(IN_CSV)
    tex = render_tex(df)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(f"saved: {OUT_TEX}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
