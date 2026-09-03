from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
CORR_PARQUET = BASE / "data" / "corr_with_marhomo1.parquet"
VARS_PARQUET = BASE / "data" / "gss_train_vars_corrected_binarized_again.parquet"
OUT = BASE / "data" / "fig_table_gen"

TOP_N = 50


def _capitalize_first(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    return s[:1].upper() + s[1:]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    corr = pd.read_parquet(CORR_PARQUET)
    corr = corr.sort_values("abs_corr", ascending=False, kind="stable").head(TOP_N)
    corr = corr.reset_index(drop=True)
    corr.insert(0, "rank", corr.index + 1)
    print(f"[A10] top-{TOP_N} correlations loaded from {CORR_PARQUET.name}")

    vm = pd.read_parquet(
        VARS_PARQUET, columns=["variable_name", "variable_label"]
    )
    label_map = dict(zip(vm["variable_name"], vm["variable_label"]))

    corr["description"] = (
        corr["variable"].map(label_map).fillna("").map(_capitalize_first)
    )

    out = corr[["rank", "variable", "description", "abs_corr"]].copy()
    out_path = OUT / "top50_samesex_corr.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[A10] saved: {out_path} ({len(out)} rows)")
    for _, r in out.head(10).iterrows():
        desc = r["description"][:60] + ("..." if len(r["description"]) > 60 else "")
        print(f"[A10]   {r['rank']:>2}. {r['variable']:<12s} "
              f"abs_corr={r['abs_corr']:.4f}  {desc}")


if __name__ == "__main__":
    main()
