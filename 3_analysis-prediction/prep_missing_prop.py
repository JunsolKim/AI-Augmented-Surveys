from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_gen"

SPLITS = ["impute", "partial", "total"]

MODELS = [
    ("alpaca", "maicomputer_alpaca-native", "Alpaca-7b"),
    ("mf",     "mf",                        "MF"),
]

RESAMPLE = {10: 1, 20: 0.888, 30: 0.777, 40: 0.666, 50: 0.555,
            60: 0.444, 70: 0.333, 80: 0.222, 90: 0.111}

PRED_TEMPLATE = "{prefix}_{split}_0_10_128_50__resample{resample}_long.parquet"


def auc_for(prefix: str, split: str, resample, gt: pd.DataFrame,
            qid_filter: set | None) -> tuple[float, int] | None:
    path = PRED / PRED_TEMPLATE.format(prefix=prefix, split=split,
                                       resample=resample)
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["yearid_id", "question_id",
                                        "response_0", "validation_0"])
    val = df.loc[df["validation_0"] == 1,
                 ["yearid_id", "question_id", "response_0"]]
    if qid_filter is not None:
        val = val[val["question_id"].isin(qid_filter)]
    val = val.merge(gt, on=["yearid_id", "question_id"], how="inner")
    val = val.dropna(subset=["binarized", "response_0"])
    if val["binarized"].nunique() < 2:
        return None
    return float(roc_auc_score(val["binarized"], val["response_0"])), int(len(val))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth from df_analysis_slim ...")
    df_slim = pd.read_pickle(DATA / "df_analysis_slim.pkl")
    df_slim = df_slim.drop_duplicates(["yearid_id", "question_id"])
    gt = (df_slim.loc[df_slim["binarized"].notna(),
                      ["yearid_id", "question_id", "binarized"]])
    print(f"  {len(gt):,} (yearid_id, question_id) ground-truth cells")

    # Retrodiction (``partial``) evaluates only questions observed in 2+ years.
    var_year = df_slim.groupby("variable")["year"].nunique()
    multi_vars = set(var_year[var_year > 1].index)
    partial_qids = set(df_slim.loc[df_slim["variable"].isin(multi_vars),
                                   "question_id"])
    print(f"  {len(partial_qids):,} question_ids eligible for partial split")

    rows = []
    for mkey, mprefix, mlabel in MODELS:
        for split in SPLITS:
            qid_filter = partial_qids if split == "partial" else None
            for pct, r in RESAMPLE.items():
                result = auc_for(mprefix, split, r, gt, qid_filter)
                if result is None:
                    print(f"  {mlabel:10s} {split:7s} {pct:2d}%: [skip]")
                    continue
                auc, n = result
                print(f"  {mlabel:10s} {split:7s} {pct:2d}%: "
                      f"AUC={auc:.4f}  n={n:,}")
                rows.append({"model": mkey, "model_label": mlabel,
                             "split": split, "missing_pct": pct,
                             "resample": r, "auc": auc, "n": n})

    out_path = OUT / "missing_prop_auc.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
