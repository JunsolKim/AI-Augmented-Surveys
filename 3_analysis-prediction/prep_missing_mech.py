from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_gen"

MECHANISMS = [
    ("MCAR", "impute"),
    ("MAR",  "mar"),
    ("MNAR", "mnar"),
]

MODELS = [
    ("alpaca", "maicomputer_alpaca-native", "Alpaca-7b"),
    ("mf",     "mf",                        "MF"),
    ("mice",   "mice_pyr",                  "MICE"),
]

PRED_TEMPLATE = "{prefix}_{split}_0_10_128_50__resample1_long.parquet"


def auc_for(prefix: str, split: str, gt: pd.DataFrame) -> tuple[float, int]:
    path = PRED / PRED_TEMPLATE.format(prefix=prefix, split=split)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path, columns=["yearid_id", "question_id",
                                        "response_0", "validation_0"])
    val = df.loc[df["validation_0"] == 1,
                 ["yearid_id", "question_id", "response_0"]]
    val = val.merge(gt, on=["yearid_id", "question_id"], how="inner")
    val = val.dropna(subset=["binarized", "response_0"])
    return float(roc_auc_score(val["binarized"], val["response_0"])), int(len(val))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth from df_analysis_slim ...")
    df_slim = pd.read_pickle(DATA / "df_analysis_slim.pkl")
    gt = (df_slim.loc[df_slim["binarized"].notna(),
                      ["yearid_id", "question_id", "binarized"]]
          .drop_duplicates(["yearid_id", "question_id"]))
    print(f"  {len(gt):,} (yearid_id, question_id) ground-truth cells")

    rows = []
    for mkey, mprefix, mlabel in MODELS:
        for mech, split in MECHANISMS:
            auc, n = auc_for(mprefix, split, gt)
            print(f"  {mlabel:10s} {mech:5s}  AUC={auc:.4f}  n={n:,}")
            rows.append({"model": mkey, "model_label": mlabel,
                         "mechanism": mech, "auc": auc, "n": n})

    out_path = OUT / "missing_mech_auc.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
