from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_gen"

ALPACA = PRED / "maicomputer_alpaca-native_partial_10_128_50__resample1_varyear.parquet"
MF     = PRED / "mf_partial_10_128_50__resample1_varyear.parquet"

N_SPLITS = 10

METHODS = [
    ("our",          "Our Method"),
    ("reg",          "Regression"),
    ("reg_logit",    "Reg+Logit"),
    ("reg_sq",       "Reg+Sq"),
    ("reg_sq_logit", "Reg+Sq+Logit"),
    ("mf",           "MF"),
]


def _build_partial_fold_map() -> dict[tuple[str, int], int]:
    """Recreate the partial-split fold assignment from
    ``2_model-finetuning/utils.load_data`` and return
    ``{(variable, year): fold_k}`` for every multi-year (var, year) cell.

    Multi-year is defined as ``count > 1`` on (question_id, year_order),
    i.e. the same val_q2 set fed to KFold with ``shuffle=True,
    random_state=42, n_splits=10`` in ``utils.load_data``.
    """
    src = DATA / "df_analysis_slim.pkl"
    if not src.exists():
        src = DATA / "df_analysis.pkl"
    df = pd.read_pickle(src).drop_duplicates(["yearid_id", "question_id"])

    # Question-year -> KFold val fold (multi-year only)
    unique_q = (df[["question_id", "year_order"]]
                .drop_duplicates()
                .reset_index(drop=True))
    unique_q["count"] = unique_q.groupby("question_id")["year_order"].transform("count")
    two = unique_q.loc[unique_q["count"] > 1].reset_index(drop=True)
    qy_to_fold: dict[tuple[int, int], int] = {}
    kfold = KFold(shuffle=True, n_splits=N_SPLITS, random_state=42)
    for k, (_, val_idx) in enumerate(kfold.split(two)):
        for idx in val_idx:
            qy_to_fold[(int(two.at[idx, "question_id"]),
                        int(two.at[idx, "year_order"]))] = k

    # (variable, year) -> (question_id, year_order) -> fold
    qy_lookup = (df[["variable", "year", "question_id", "year_order"]]
                 .drop_duplicates(["variable", "year"]))
    vy_fold: dict[tuple[str, int], int] = {}
    for _, r in qy_lookup.iterrows():
        f = qy_to_fold.get((int(r["question_id"]), int(r["year_order"])))
        if f is not None:
            vy_fold[(str(r["variable"]), int(r["year"]))] = f
    return vy_fold


def _fit_loo_regressions(vy: pd.DataFrame,
                         fold_map: dict[tuple[str, int], int]) -> pd.DataFrame:
    """Per-variable, **fold-aware** LOO predictions for the four regression
    baselines.

    For each held-out (V, Y_te) with fold ``k_te = fold_map[(V, Y_te)]``,
    the regression is fit on (V, Y') only when ``fold_map[(V, Y')] != k_te``
    — i.e. on the same set of years the LLM saw during training for fold
    ``k_te``. Linear needs >=2 such training years; quadratic needs >=3.
    Cells failing those thresholds get NaN for that method (the breakdown
    rows keep these as N/A; only the "All" aggregate row applies a
    fallback chain to avoid selection bias).

    An additional ``intercept_only`` column is always emitted: the
    fold-aware mean of obs_wavg over the training years (= mean of o_tr).
    This is the deepest fallback the ``_summarise`` "All" aggregate uses
    when even the linear fit is underdetermined (n_tr < 2).

    Cells with no fold entry are skipped.
    """
    n_skipped_no_fold = 0
    n_empty_train = 0
    recs = []
    for var, g in vy.groupby("variable"):
        g = g.sort_values("year").reset_index(drop=True)
        years = g["year"].to_numpy(dtype=float)
        obs = g["obs_wavg"].to_numpy()
        folds = np.array([fold_map.get((var, int(y)), -1) for y in years],
                         dtype=int)
        n = len(g)
        if n < 2:
            continue
        for i in range(n):
            y_te = years[i]
            k_te = folds[i]
            if k_te < 0:
                n_skipped_no_fold += 1
                continue
            # Fold-aware train mask: exclude every (V, Y') that was in the
            # same fold's val as Y_te (this also drops Y_te itself).
            mask_tr = folds != k_te
            y_tr = years[mask_tr]
            o_tr = obs[mask_tr]
            n_tr = len(y_tr)
            intercept_only = float(np.mean(o_tr)) if n_tr >= 1 else np.nan
            if n_tr == 0:
                n_empty_train += 1
                recs.append({"variable": var, "year": int(y_te),
                             "reg": np.nan, "reg_logit": np.nan,
                             "reg_sq": np.nan, "reg_sq_logit": np.nan,
                             "intercept_only": np.nan})
                continue

            if n_tr >= 2:
                X1 = sm.add_constant(y_tr)
                lin = sm.OLS(o_tr, X1).fit().predict([1.0, y_te])[0]
                try:
                    glm1 = (sm.GLM(o_tr, X1, family=sm.families.Binomial())
                            .fit(disp=0).predict([1.0, y_te])[0])
                except Exception:
                    glm1 = np.nan
            else:
                lin, glm1 = np.nan, np.nan

            if n_tr >= 3:
                X2 = sm.add_constant(np.column_stack([y_tr, y_tr ** 2]))
                sq = sm.OLS(o_tr, X2).fit().predict([1.0, y_te, y_te ** 2])[0]
                try:
                    glm2 = (sm.GLM(o_tr, X2, family=sm.families.Binomial())
                            .fit(disp=0).predict([1.0, y_te, y_te ** 2])[0])
                except Exception:
                    glm2 = np.nan
            else:
                sq, glm2 = np.nan, np.nan

            recs.append({"variable": var, "year": int(y_te),
                         "reg": lin, "reg_logit": glm1,
                         "reg_sq": sq, "reg_sq_logit": glm2,
                         "intercept_only": intercept_only})
    if n_skipped_no_fold:
        print(f"  WARN: {n_skipped_no_fold} test cells skipped (no fold mapping)")
    if n_empty_train:
        print(f"  NOTE: {n_empty_train} test cells had empty fold-aware train set")
    return pd.DataFrame(recs)


def _apply_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the regression-baseline NaNs with fallback rules so
    the "All" aggregate isn't biased by selection on the easy cells:

      n_tr = 1 (Reg / Reg+Logit underdetermined)
        → intercept_only (= the single training year's obs_wavg)
      n_tr = 2 (Reg+Sq / Reg+Sq+Logit quadratic underdetermined)
        → Reg (linear OLS) — itself already filled with intercept_only
          for the n_tr=1 cells via the chain above

    Implemented as ``combine_first`` chains: each method takes its own
    value when present, otherwise the next-simpler model in the chain.
    """
    if "intercept_only" not in df.columns:
        return df
    df = df.copy()
    ic = df["intercept_only"]
    df["reg"]          = df["reg"].combine_first(ic)
    df["reg_logit"]    = df["reg_logit"].combine_first(ic)
    df["reg_sq"]       = df["reg_sq"].combine_first(df["reg"])
    df["reg_sq_logit"] = df["reg_sq_logit"].combine_first(df["reg"])
    return df


def _summarise(df: pd.DataFrame, scenario: str, group: str,
               use_fallback: bool = False) -> dict:
    if use_fallback:
        df = _apply_fallback(df)
    out: dict = {
        "scenario": scenario, "group": group,
        "n_var_year": int(len(df)),
        "n_var": int(df["variable"].nunique()),
    }
    for col, _ in METHODS:
        d = df.dropna(subset=[col, "obs_wavg"])
        if len(d) < 3:
            out[f"{col}_r"] = np.nan
            out[f"{col}_mae"] = np.nan
            continue
        out[f"{col}_r"] = float(d[col].corr(d["obs_wavg"], method="spearman"))
        out[f"{col}_mae"] = float((d[col] - d["obs_wavg"]).abs().mean())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    if not ALPACA.exists():
        raise SystemExit(f"missing Alpaca varyear: {ALPACA}")

    vy_alp = pd.read_parquet(ALPACA).dropna(subset=["GLM_predicted", "obs_wavg"])
    vy_mf = (pd.read_parquet(MF).dropna(subset=["GLM_predicted", "obs_wavg"])
             if MF.exists() else pd.DataFrame())
    print(f"alpaca varyear: {len(vy_alp):,} (var, year) rows")
    print(f"mf     varyear: {len(vy_mf):,} (var, year) rows")

    print("building partial-split fold map ...")
    fold_map = _build_partial_fold_map()
    print(f"  fold map covers {len(fold_map):,} (var, year) cells")
    # Quick sanity: the partial varyear's (var, year) set should be a
    # subset of fold_map.
    n_in = sum((str(v), int(y)) in fold_map
               for v, y in zip(vy_alp["variable"], vy_alp["year"]))
    print(f"  fold map covers {n_in:,} / {len(vy_alp):,} varyear rows")

    print("fitting per-variable, fold-aware LOO regressions ...")
    reg_df = _fit_loo_regressions(vy_alp, fold_map)
    print(f"  regression preds: {len(reg_df):,} rows")

    merged = (vy_alp.rename(columns={"GLM_predicted": "our"})
              [["variable", "year", "obs_wavg", "our"]]
              .merge(reg_df, on=["variable", "year"], how="left"))
    if len(vy_mf):
        merged = merged.merge(
            vy_mf.rename(columns={"GLM_predicted": "mf"})
            [["variable", "year", "mf"]],
            on=["variable", "year"], how="left")
    else:
        merged["mf"] = np.nan

    # --- Per-variable features used for grouping ---
    by_var = merged.groupby("variable")
    n_years = by_var["year"].nunique()
    volatility = by_var["obs_wavg"].std().fillna(0)
    obs_years_map = by_var["year"].apply(lambda s: np.asarray(sorted(set(s))))

    def row_dist(var: str, yr: int) -> float:
        ys = obs_years_map.get(var, np.array([]))
        if len(ys) <= 1:
            return np.nan
        return float(np.min(np.abs(ys[ys != yr] - yr)))

    # n_train_years = (years available for this variable) - 1 (held-out year)
    merged["n_train_years"] = merged["variable"].map(n_years) - 1
    merged["volatility"] = merged["variable"].map(volatility)
    merged["t_distance"] = [row_dist(v, y) for v, y in
                            zip(merged["variable"], merged["year"])]

    # Volatility threshold: mean of the per-variable std.
    vol_threshold = float(merged["volatility"].mean())

    rows = [_summarise(merged, "All", "All", use_fallback=True)]
    for lab, mask in [("year=1", merged["n_train_years"] == 1),
                      ("year=2", merged["n_train_years"] == 2),
                      ("year>2", merged["n_train_years"] > 2)]:
        rows.append(_summarise(merged[mask], "Sparsity", lab))
    for lab, mask in [("high", merged["volatility"] >= vol_threshold),
                      ("low",  merged["volatility"] <  vol_threshold)]:
        rows.append(_summarise(merged[mask], "Volatility", lab))
    for lab, mask in [("dist>=3", merged["t_distance"] >= 3),
                      ("dist<3",  merged["t_distance"] < 3)]:
        rows.append(_summarise(merged[mask], "Temporal distance", lab))

    for r in rows:
        cells = " ".join(
            f"{lab}={r.get(f'{k}_r', float('nan')):.3f}/"
            f"{r.get(f'{k}_mae', float('nan')):.3f}"
            for k, lab in METHODS)
        print(f"[A8] {r['scenario']:18s} {r['group']:8s}: {cells}  "
              f"n={r['n_var_year']}")

    out_path = OUT / "perfbygroup_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
