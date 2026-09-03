from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


np.random.seed(1234)

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
PRED = BASE / "data" / "predictions"
OUT = BASE / "data" / "fig_table_gen"

TASKS = [
    ("impute", "Missing Data Imputation"),
    ("partial", "Retrodiction"),
    ("total", "Unasked Opinion Prediction"),
]
ROC_N_POINTS = 500


def downsample_roc(
    fpr: np.ndarray, tpr: np.ndarray, n: int = ROC_N_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    k = len(fpr)
    if k <= n:
        return fpr, tpr
    step = max(1, k // n)
    idx = np.arange(0, k, step)
    if idx[-1] != k - 1:
        idx = np.concatenate([idx, [k - 1]])
    return fpr[idx], tpr[idx]


def build_roc(roc_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roc_rows, auc_rows = [], []
    for split, task_label in TASKS:
        obs_col, pred_col = f"{split}_obs", f"{split}_pred"
        mask = roc_df[obs_col].notna()
        y_true = roc_df.loc[mask, obs_col].astype(int).values
        y_score = roc_df.loc[mask, pred_col].values
        n = int(mask.sum())

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = float(roc_auc_score(y_true, y_score))
        fpr_ds, tpr_ds = downsample_roc(fpr, tpr)

        roc_rows.append(pd.DataFrame({
            "task": task_label,
            "split": split,
            "fpr": fpr_ds,
            "tpr": tpr_ds,
        }))
        auc_rows.append({"task": task_label, "split": split, "auc": auc, "n": n})
        print(f"  {split:>7}: n={n:>8,}  AUC={auc:.4f}")

    return pd.concat(roc_rows, ignore_index=True), pd.DataFrame(auc_rows)


def build_scatter() -> pd.DataFrame:
    parts = []
    for split, task_label in TASKS:
        fn = PRED / (
            f"maicomputer_alpaca-native_{split}_10_128_50__resample1_varyear.parquet"
        )
        df = pd.read_parquet(fn)
        out = pd.DataFrame({
            "task": task_label,
            "split": split,
            "variable": df["variable"].values,
            "year": df["year"].astype(int).values,
            "pred_mean": df["GLM_predicted"].astype(float).values,
            "obs_mean": df["obs_wavg"].astype(float).values,
            "n_val": df["n_val"].astype(int).values,
            "sw_val": df["sw_val"].astype(float).values,
        })
        out["diff_mean"] = out["pred_mean"] - out["obs_mean"]
        parts.append(out)

        r = np.corrcoef(out["pred_mean"], out["obs_mean"])[0, 1]
        pct3 = (out["diff_mean"].abs() < 0.03).mean() * 100
        pct5 = (out["diff_mean"].abs() < 0.05).mean() * 100
        print(
            f"  {split:>7}: rows={len(out):>5,}  "
            f"R={r:.4f}  %±3%={pct3:.1f}  %±5%={pct5:.1f}"
        )

    return pd.concat(parts, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Panel A: ROC curves")
    roc_df = pd.read_parquet(PROC / "figure4_roc_data.parquet")
    roc_plot_df, auc_df = build_roc(roc_df)

    roc_out = OUT / "modelperformance_roc_downsampled.parquet"
    auc_out = OUT / "modelperformance_roc_auc.csv"
    roc_plot_df.to_parquet(roc_out, index=False)
    auc_df.to_csv(auc_out, index=False)
    print(f"  saved: {roc_out}")
    print(f"  saved: {auc_out}")

    print("\nPanels B–D: varyear scatter")
    scatter_df = build_scatter()
    scatter_out = OUT / "modelperformance_scatter.parquet"
    scatter_df.to_parquet(scatter_out, index=False)
    print(f"  saved: {scatter_out}")


if __name__ == "__main__":
    main()
