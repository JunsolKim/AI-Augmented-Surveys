from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
IN_CSV = (BASE / "data" / "fig_table_gen" / "alpaca_vs_mf_groupauc.csv")
OUT_DIR = BASE / "output"

GROUP_ORDER = ["sex", "age_group", "period_group", "race",
               "region4", "degree", "income_q5", "partyid7"]
GROUP_LABEL = {
    "sex": "Sex", "age_group": "Age", "period_group": "Period",
    "race": "Race", "region4": "Region", "degree": "Education",
    "income_q5": "Income", "partyid7": "Party ID",
}
LEVEL_ORDER = {
    "sex":          ["male", "female"],
    "age_group":    ["18-29", "30-44", "45-59", "60-74", "75-89"],
    "period_group": ["1970s", "1980s", "1990s", "2000s", "2010s"],
    "race":         ["White", "Black", "Other"],
    "region4":      ["NE", "MW", "S", "W"],
    "degree":       ["below_hs", "hs", "some_college", "bs", "bs_above"],
    "income_q5":    ["Income Q1", "Income Q2", "Income Q3",
                     "Income Q4", "Income Q5", "Income NA"],
    "partyid7":     ["independent", "strong democrat", "weak democrat",
                     "lean toward democrat", "lean toward republican",
                     "weak republican", "strong republican", "other party"],
}
LEVEL_PRETTY = {
    "sex":          {"male": "Male", "female": "Female"},
    "age_group":    {"18-29": "18--29", "30-44": "30--44", "45-59": "45--59",
                     "60-74": "60--74", "75-89": "75--89"},
    "period_group": {"1970s": "1970s", "1980s": "1980s", "1990s": "1990s",
                     "2000s": "2000s", "2010s": "2010s"},
    "race":         {"White": "White", "Black": "Black", "Other": "Other"},
    "region4":      {"NE": "Northeast", "MW": "Midwest",
                     "S": "South", "W": "West"},
    "degree":       {"below_hs": "Less than high school",
                     "hs": "High school graduate",
                     "some_college": "Some college",
                     "bs": "Bachelor's",
                     "bs_above": "Graduate"},
    "income_q5":    {"Income Q1": "Q1 (lowest)", "Income Q2": "Q2",
                     "Income Q3": "Q3", "Income Q4": "Q4",
                     "Income Q5": "Q5 (highest)", "Income NA": "Missing"},
    "partyid7":     {"independent": "Independent",
                     "strong democrat": "Strong Democrat",
                     "weak democrat": "Weak Democrat",
                     "lean toward democrat": "Lean Democrat",
                     "lean toward republican": "Lean Republican",
                     "weak republican": "Weak Republican",
                     "strong republican": "Strong Republican",
                     "other party": "Something else"},
}

TASKS = [
    ("Missing Data Imputation",
     "table_subgroup_delta_imputation.tex",
     r"Per-subgroup group-level AUC, Alpaca-7b vs.\ matrix factorization, on the missing-data imputation task",
     "tab:subgroup_groupauc_imputation"),
    ("Retrodiction",
     "table_subgroup_delta_retrodiction.tex",
     r"Per-subgroup group-level AUC, Alpaca-7b vs.\ matrix factorization, on the retrodiction task",
     "tab:subgroup_groupauc_retrodiction"),
]


def per_subgroup(df_task: pd.DataFrame) -> pd.DataFrame:
    """Long → one row per (demo_group, subgroup) with AUCs for both models."""
    wide = df_task.pivot_table(
        index=["demo_group", "subgroup", "n_respondents", "n_obs"],
        columns="model", values="auc", aggfunc="first"
    ).reset_index().rename(
        columns={"Alpaca-7b": "auc_alpaca", "MF": "auc_mf"})
    wide["delta"] = wide["auc_alpaca"] - wide["auc_mf"]

    # Order rows according to GROUP_ORDER × LEVEL_ORDER.
    rows = []
    for g in GROUP_ORDER:
        sub = wide[wide["demo_group"] == g]
        for lvl in LEVEL_ORDER[g]:
            r = sub[sub["subgroup"] == lvl]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({
                "demo_group":    GROUP_LABEL[g],
                "subgroup":      LEVEL_PRETTY[g][lvl],
                "n_respondents": int(r["n_respondents"]),
                "auc_alpaca":    float(r["auc_alpaca"]),
                "auc_mf":        float(r["auc_mf"]),
                "delta":         float(r["delta"]),
            })
    return pd.DataFrame(rows)


def render_tex(df: pd.DataFrame, caption: str, label: str) -> str:
    lines: list[str] = [
        r"\begin{table}[H]",
        rf"  \caption{{\textbf{{Table XX.}} {caption}.}}",
        rf"  \label{{{label}}}",
        r"  \centering",
        r"  \small",
        r"  \renewcommand{\arraystretch}{0.92}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        (r"    \textbf{Group} & \textbf{Subgroup} & $N$ "
         r"& \textbf{Alpaca-7b} & \textbf{MF} "
         r"& $\mathbf{\Delta}$ \textbf{(Alpaca $-$ MF)} \\"),
        r"     &  &  & (AUC) & (AUC) & (AUC) \\",
        r"    \midrule",
    ]

    groups = df["demo_group"].drop_duplicates().tolist()
    for gi, g in enumerate(groups):
        sub = df[df["demo_group"] == g]
        n_rows = len(sub)
        for ri, (_, r) in enumerate(sub.iterrows()):
            first_cell = (rf"\multirow{{{n_rows}}}{{*}}{{\textbf{{{g}}}}}"
                          if ri == 0 else "")
            lines.append(
                f"    {first_cell} & {r['subgroup']} "
                f"& {int(r['n_respondents']):,} "
                f"& {r['auc_alpaca']:.3f} "
                f"& {r['auc_mf']:.3f} "
                rf"& \textbf{{{r['delta']:+.3f}}} \\"
            )
        if gi < len(groups) - 1:
            lines.append(r"    \midrule")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"")
    lines.append(r"  \medskip")
    lines.append(
        r"  \parbox{\linewidth}{\footnotesize \textit{Notes.} $\Delta$ "
        r"(Alpaca $-$ MF) is the per-subgroup difference in group-level "
        r"AUC, where group-level AUC is computed by pooling held-out "
        r"predictions within each subgroup. Bolded $\Delta$ values "
        r"highlight where the proposed approach improves over the "
        r"matrix-factorization baseline.}"
    )
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.read_csv(IN_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_label, tex_name, caption, label in TASKS:
        sub = df[df["task"] == task_label]
        per_sub = per_subgroup(sub)
        tex = render_tex(per_sub, caption, label)
        out_tex = OUT_DIR / tex_name
        out_tex.write_text(tex)
        print(f"saved: {out_tex}  ({len(per_sub)} subgroups)")


if __name__ == "__main__":
    main()
