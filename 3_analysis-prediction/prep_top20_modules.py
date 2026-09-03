import json
import pickle
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
VARS_PARQUET = BASE / "data" / "gss_train_vars_corrected_binarized_again.parquet"
MODULE_PKL = BASE / "data" / "module_dict.pkl"
DF_SLIM_PKL = BASE / "data" / "df_analysis_slim.pkl"
OUT = BASE / "data" / "fig_table_gen"

EXCLUDE_MODULES = {"Core"}
TOP_N = 20
REP_N = 3


def _load_module_map() -> dict[str, str]:
    """Return a flat {variable_name: module_label} mapping.

    Accepts either a flat var->module dict or a module->[vars] dict
    (invert the latter). Falls back to
    df_analysis_slim.pkl's ``module`` column if module_dict.pkl is missing.
    """
    if MODULE_PKL.exists():
        with open(MODULE_PKL, "rb") as f:
            md = pickle.load(f)
        if not isinstance(md, dict) or not md:
            raise ValueError(f"unexpected module_dict shape: {type(md)}")
        any_val = next(iter(md.values()))
        if isinstance(any_val, (list, set, tuple)):
            return {v: m for m, vs in md.items() for v in vs}
        return dict(md)

    print(f"[A9] module_dict.pkl not found, falling back to {DF_SLIM_PKL.name}")
    df = pd.read_pickle(DF_SLIM_PKL)
    return dict(zip(df["variable"], df["module"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    vm = pd.read_parquet(VARS_PARQUET)
    vm = vm[vm["use_variable_final"] == 1][["variable_name", "variable_label"]].copy()
    print(f"[A9] analytic variables: {len(vm):,}")

    mod_map = _load_module_map()
    vm["module"] = vm["variable_name"].map(mod_map)
    n_mapped = vm["module"].notna().sum()
    print(f"[A9] variables with a module: {n_mapped:,} / {len(vm):,}")

    vm = vm.dropna(subset=["module"])
    vm = vm[~vm["module"].isin(EXCLUDE_MODULES)].copy()
    print(f"[A9] after excluding {sorted(EXCLUDE_MODULES)}: {len(vm):,} rows, "
          f"{vm['module'].nunique()} modules")

    def _rep_vars(group: pd.DataFrame) -> str:
        seen: list[dict] = []
        for _, r in group.iterrows():
            if len(seen) >= REP_N:
                break
            name = str(r["variable_name"])
            if any(d["variable"] == name for d in seen):
                continue
            label = r["variable_label"] if pd.notna(r["variable_label"]) else ""
            seen.append({"variable": name, "label": str(label).strip()})
        return json.dumps(seen, ensure_ascii=False)

    agg = (
        vm.groupby("module", sort=False)
        .apply(lambda g: pd.Series({
            "n_variables": int(g["variable_name"].nunique()),
            "rep_vars_json": _rep_vars(g),
        }))
        .reset_index()
    )
    agg = agg.sort_values("n_variables", ascending=False, kind="stable").head(TOP_N)
    agg = agg.reset_index(drop=True)
    agg.insert(0, "rank", agg.index + 1)

    out_path = OUT / "top20_modules.parquet"
    agg.to_parquet(out_path, index=False)
    print(f"[A9] saved: {out_path} ({len(agg)} rows)")
    for _, r in agg.iterrows():
        rv = json.loads(r["rep_vars_json"])
        rv_str = ", ".join(d["variable"] for d in rv)
        print(f"[A9]   {r['rank']:>2}. {r['module']:<45s} n={r['n_variables']:>4}  [{rv_str}]")


if __name__ == "__main__":
    main()
