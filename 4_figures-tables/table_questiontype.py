from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PARQUET = REPO / "data" / "gss_train_vars_corrected_binarized_again.parquet"
OUT_TEX = REPO / "output" / "table_question_distribution.tex"


# Display labels for each raw LLM category. Categories absent from the final
# analytic set are silently skipped.
CATEGORY_LABELS = {
    "Attitudinal/Opinion Questions": "Attitudinal/Opinion Questions",
    "Behavioral Questions": "Behavioral Questions",
    "Objective/Knowledge/Factual Questions": "Objective/Knowledge/Factual Questions",
    "Questions about Other People": (
        "Questions about Other People (e.g., relationship and household rosters)"
    ),
    "Demographic Questions": "Demographic Questions",
    "Open-ended Questions": "Open-ended Questions",
    "Derived/Computed/Condition Questions": "Derived/Computed/Condition Questions",
    "Metadata/Paradata": "Metadata/Paradata",
    "Miscellaneous Questions": "Miscellaneous Questions",
}


def build_rows(df: pd.DataFrame) -> tuple[list[tuple[str, int, float]], int]:
    """Return (rows, total), where each row is (label, count, percent)."""
    total = len(df)
    counts = df["category_original"].value_counts(dropna=False).to_dict()

    rows = []
    seen = 0
    for cat_raw, label in CATEGORY_LABELS.items():
        n = int(counts.get(cat_raw, 0))
        if n == 0:
            continue
        rows.append((label, n, n / total * 100))
        seen += n

    assert seen == total, f"Category counts sum to {seen}, expected {total}"

    rows.sort(key=lambda r: -r[1])
    return rows, total


def render_tex(rows: list[tuple[str, int, float]], total: int) -> str:
    label_w = max(len(r[0]) for r in rows)
    label_w = max(label_w, len("Total Questions"))

    body = []
    for label, n, pct in rows:
        body.append(f"{label:<{label_w}} & {n:>5} & {pct:>6.2f}\\% \\\\")
    total_label = r"\textbf{Total Questions}"
    pad = " " * max(0, label_w - len("Total Questions"))
    body.append(f"{total_label}{pad} & {total:>5} & 100.00\\% \\\\")

    return "\n".join([
        r"\begin{table}[htp]",
        r"\centering",
        r"\caption{Distribution of Question Types}",
        r"\label{tab:question_distribution}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Variable Category} & \textbf{Frequency} & \textbf{Proportion} \\",
        r"\midrule",
        *body[:-1],
        r"\midrule",
        body[-1],
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df = df[df["use_variable_final"] == 1].copy()

    rows, total = build_rows(df)
    tex = render_tex(rows, total)

    OUT_TEX.parent.mkdir(exist_ok=True)
    OUT_TEX.write_text(tex)

    print(f"saved: {OUT_TEX}")
    print(f"total final analytic variables: {total}")
    for label, n, pct in rows:
        print(f"  {n:>5}  {pct:>6.2f}%  {label}")


if __name__ == "__main__":
    main()
