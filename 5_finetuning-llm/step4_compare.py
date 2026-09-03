from __future__ import annotations

import sys
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, f1_score, accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2_model-finetuning"))

REPL = HERE.parent
PRED_DIR = REPL / "data" / "predictions"
LOGS_DIR = HERE / "logs"

BASELINE_PREFIX = "maicomputer_alpaca-native"


def _pred_path(model_name: str, split: str, k: int) -> Path:
    return PRED_DIR / f"{model_name}_{split}_{k}_10_128_50__resample1_long.parquet"


def _eval_overall(df_val: pd.DataFrame, pred_col: str):
    y = df_val["binarized"].to_numpy().astype(int)
    p = df_val[pred_col].to_numpy()
    yhat = (p >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "bce": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "acc": float(accuracy_score(y, yhat)),
        "f1":  float(f1_score(y, yhat, zero_division=0)),
        "n":   int(len(df_val)),
    }


def _per_question_auc(df_val: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for q, g in df_val.groupby("question_id"):
        y = g["binarized"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        rows.append({"question_id": int(q), "auc": float(roc_auc_score(y, g[pred_col].to_numpy())),
                     "n": int(len(g))})
    return pd.DataFrame(rows)


def _load_with_labels(model_name: str, split: str, k: int):
    """Load long parquet, attach `binarized` from df_analysis, and keep only val rows."""
    path = _pred_path(model_name, split, k)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = df[df[f"validation_{k}"] == 1].copy()
    # attach binarized via df_analysis
    from utils import load_data
    _df_a, _train, val = load_data(split_type=split, k=k, n_splits=10,
                                    resample=1, use_demo=False, use_poli=True)
    df = df.merge(val[["yearid_id", "question_id", "binarized"]],
                  on=["yearid_id", "question_id"], how="inner")
    return df


def main(
    k: int = 0,
    n: int = 200_000,
    lora_r: int = 16,
    variants: str = "all-linear,qv",
    splits: str = "impute,partial,total",
):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in variants.split(",")]
    splits = [s.strip() for s in splits.split(",")]

    rows = []
    per_q_frames = []
    for split in splits:
        baseline_df = _load_with_labels(BASELINE_PREFIX, split, k)
        if baseline_df is None:
            print(f"[stage4] missing baseline for split={split} — skipping")
            continue
        m = _eval_overall(baseline_df, f"response_{k}")
        rows.append({"split": split, "model": "baseline", **m})
        bq = _per_question_auc(baseline_df, f"response_{k}").rename(columns={"auc": "auc_baseline"})

        for var in variants:
            mname = f"alpaca-ft-intersection-k{k}-{var}-r{lora_r}-n{n}"
            df = _load_with_labels(mname, split, k)
            if df is None:
                print(f"[stage4] missing predictions for split={split} variant={var} ({mname})")
                continue
            m = _eval_overall(df, f"response_{k}")
            rows.append({"split": split, "model": var, **m})
            fq = _per_question_auc(df, f"response_{k}").rename(columns={"auc": f"auc_{var}"})
            bq = bq.merge(fq[["question_id", f"auc_{var}"]], on="question_id", how="outer")

        bq["split"] = split
        per_q_frames.append(bq)

    overall = pd.DataFrame(rows)
    print("\n=== overall ===")
    print(overall.to_string(index=False))
    overall.to_csv(LOGS_DIR / f"stage4_compare_k{k}_n{n}.csv", index=False)
    print(f"saved -> {LOGS_DIR / f'stage4_compare_k{k}_n{n}.csv'}")

    if per_q_frames:
        per_q = pd.concat(per_q_frames, axis=0, ignore_index=True)
        per_q.to_csv(LOGS_DIR / f"stage4_per_question_k{k}_n{n}.csv", index=False)
        print(f"saved -> {LOGS_DIR / f'stage4_per_question_k{k}_n{n}.csv'}")

        # quick scatter plot per variant
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n_splits_p = len(splits)
            fig, axes = plt.subplots(len(variants), n_splits_p,
                                      figsize=(4 * n_splits_p, 4 * len(variants)),
                                      squeeze=False)
            for i, var in enumerate(variants):
                for j, split in enumerate(splits):
                    ax = axes[i][j]
                    sub = per_q[per_q.split == split]
                    if f"auc_{var}" not in sub.columns:
                        ax.set_visible(False); continue
                    ax.scatter(sub["auc_baseline"], sub[f"auc_{var}"], s=3, alpha=0.4)
                    lim = [0.4, 1.0]
                    ax.plot(lim, lim, color="red", linewidth=0.8)
                    ax.set_xlim(lim); ax.set_ylim(lim)
                    ax.set_xlabel("baseline AUC"); ax.set_ylabel(f"{var} AUC")
                    ax.set_title(f"{split}  ({len(sub)} q)")
            fig.tight_layout()
            out_pdf = LOGS_DIR / f"stage4_delta_auc_k{k}_n{n}.pdf"
            fig.savefig(out_pdf)
            print(f"saved -> {out_pdf}")
        except Exception as e:
            print(f"[stage4] plot failed: {e}")


if __name__ == "__main__":
    fire.Fire(main)
