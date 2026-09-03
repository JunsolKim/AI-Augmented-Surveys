from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_gen"

MODELS = [
    ("alpaca",  "maicomputer_alpaca-native", "Alpaca-7b",            True),
    ("gptj",    "EleutherAI_gpt-j-6B",       "GPT-J-6b",             True),
    ("roberta", "roberta-large",             "RoBERTa-large",        True),
    ("mf",      "mf",                        "Matrix Factorization", False),
    ("mice",    "mice_pyr",                  "MICE",                 False),
]

SCENARIOS = [
    ("impute",  "Missing data imputation"),
    ("partial", "Retrodiction"),
    ("total",   "Unasked opinion prediction"),
]

META_COLS = {"yearid_id", "yearid", "year", "id",
             "wtssall", "wtssnr", "wtssnrps", "polviews",
             "sampcode", "realinc", "income_category"}


def metrics_from_wide(wide_path: Path, split_type: str,
                      multi_year_vars: set[str]) -> tuple[float, float, float, int]:
    """Pool predictions over all ``{var}`` columns that have a matching
    ``{var}_obs_bin`` ground-truth column, then compute AUC / Acc / F1."""
    df = pd.read_parquet(wide_path)
    pred_cols = [c for c in df.columns
                 if f"{c}_obs_bin" in df.columns and c not in META_COLS]
    if split_type == "partial":
        pred_cols = [c for c in pred_cols if c in multi_year_vars]

    preds, obs = [], []
    for v in pred_cols:
        mask = df[f"{v}_obs_bin"].notna()
        if mask.any():
            preds.append(df.loc[mask, v].to_numpy())
            obs.append(df.loc[mask, f"{v}_obs_bin"].to_numpy())
    pred = np.concatenate(preds)
    y = np.concatenate(obs).astype(np.int32)
    yhat = (pred >= 0.5).astype(np.int32)
    return (float(roc_auc_score(y, pred)),
            float(accuracy_score(y, yhat)),
            float(f1_score(y, yhat, zero_division=0)),
            int(len(pred)))


def metrics_from_mice_folds(split_type: str,
                            gt: pd.DataFrame,
                            qid_to_var: dict[int, str],
                            multi_year_vars: set[str]
                            ) -> tuple[float, float, float, int] | None:
    """Concatenate ``validation_k == 1`` cells across all 10 MICE py_r folds."""
    parts = []
    for k in range(10):
        path = PRED / f"mice_pyr_{split_type}_{k}_10_128_50__resample1_long.parquet"
        if not path.exists():
            print(f"  MICE fold {k} missing: {path.name}")
            return None
        df = pd.read_parquet(path, columns=["yearid_id", "question_id",
                                            f"validation_{k}", f"response_{k}"])
        v = df.loc[df[f"validation_{k}"] == 1,
                   ["yearid_id", "question_id", f"response_{k}"]]
        v = v.rename(columns={f"response_{k}": "pred"})
        parts.append(v)
    pooled = pd.concat(parts, ignore_index=True)
    pooled = pooled.merge(gt, on=["yearid_id", "question_id"], how="inner")
    pooled = pooled.dropna(subset=["binarized", "pred"])

    if split_type == "partial":
        pooled["variable"] = pooled["question_id"].map(qid_to_var)
        pooled = pooled[pooled["variable"].isin(multi_year_vars)]

    y = pooled["binarized"].to_numpy(dtype=np.int32)
    pred = pooled["pred"].to_numpy(dtype=np.float64)
    yhat = (pred >= 0.5).astype(np.int32)
    return (float(roc_auc_score(y, pred)),
            float(accuracy_score(y, yhat)),
            float(f1_score(y, yhat, zero_division=0)),
            int(len(pooled)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading df_analysis_slim for ground truth and variable metadata ...")
    df_slim = pd.read_pickle(DATA / "df_analysis_slim.pkl")
    gt = (df_slim.loc[df_slim["binarized"].notna(),
                      ["yearid_id", "question_id", "binarized"]]
          .drop_duplicates(["yearid_id", "question_id"]))
    qid_to_var = dict(df_slim[["question_id", "variable"]]
                      .drop_duplicates().to_numpy())
    var_year_counts = df_slim.groupby("variable")["year"].nunique()
    multi_year_vars = set(var_year_counts[var_year_counts > 1].index)
    print(f"  {len(qid_to_var):,} variables, "
          f"{len(multi_year_vars):,} multi-year")

    rows = []
    for mkey, mprefix, mlabel, has_total in MODELS:
        for skey, slabel in SCENARIOS:
            if skey == "total" and not has_total:
                rows.append({"model": mkey, "model_label": mlabel,
                             "scenario": skey, "scenario_label": slabel,
                             "auc": None, "acc": None,
                             "f1": None, "n": None})
                continue

            if mkey == "mice":
                print(f"{mlabel:22s} x {skey:8s}: pooling 10 MICE py_r folds")
                m = metrics_from_mice_folds(skey, gt, qid_to_var,
                                            multi_year_vars)
            else:
                wide = PRED / (f"{mprefix}_{skey}_10_128_50"
                               f"__resample1_wide.parquet")
                if not wide.exists():
                    print(f"{mlabel:22s} x {skey:8s}: WIDE NOT FOUND ({wide.name})")
                    m = None
                else:
                    print(f"{mlabel:22s} x {skey:8s}: {wide.name}")
                    m = metrics_from_wide(wide, skey, multi_year_vars)

            if m is None:
                rows.append({"model": mkey, "model_label": mlabel,
                             "scenario": skey, "scenario_label": slabel,
                             "auc": None, "acc": None,
                             "f1": None, "n": None})
            else:
                auc, acc, f1, n = m
                print(f"    AUC={auc:.4f}  Acc={acc:.4f}  "
                      f"F1={f1:.4f}  N={n:,}")
                rows.append({"model": mkey, "model_label": mlabel,
                             "scenario": skey, "scenario_label": slabel,
                             "auc": auc, "acc": acc, "f1": f1, "n": n})

    df = pd.DataFrame(rows)
    out_path = OUT / "modelcomparison_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
