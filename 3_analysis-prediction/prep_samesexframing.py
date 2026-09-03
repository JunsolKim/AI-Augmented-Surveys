from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


np.random.seed(1234)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PRED = DATA / "predictions"
OUT = DATA / "fig_table_gen"

MODEL_STEM = "maicomputer_alpaca-native"
SPLIT = "partial"
VARS = ["marhomo1", "marhomo"]


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log((1.0 / (x + 1e-8)) - 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_varyear_glm(varyear_path: Path) -> tuple[float, float, int]:
    vy = pd.read_parquet(varyear_path).dropna(subset=["pred_logit_wavg", "obs_wavg"])
    X = sm.add_constant(vy["pred_logit_wavg"].to_numpy())
    glm = sm.GLM(vy["obs_wavg"].to_numpy(), X, family=sm.families.Binomial()).fit()
    return float(glm.params[0]), float(glm.params[1]), int(len(vy))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    vy_path = PRED / f"{MODEL_STEM}_{SPLIT}_10_128_50__resample1_varyear.parquet"
    wide_path = PRED / f"{MODEL_STEM}_{SPLIT}_10_128_50__resample1_wide.parquet"

    b0, b1, n_train = fit_varyear_glm(vy_path)
    print(f"varyear GLM: β₀={b0:.4f}  β₁={b1:.4f}  (n={n_train:,})")

    wide = pd.read_parquet(wide_path, columns=["yearid_id", "year"] + VARS)
    print(f"wide parquet: {len(wide):,} respondents")

    raw1 = wide["marhomo1"].to_numpy()
    raw2 = wide["marhomo"].to_numpy()
    lg1 = _logit(raw1)
    lg2 = _logit(raw2)
    mask = np.isfinite(lg1) & np.isfinite(lg2)
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        print(f"  dropped {n_dropped:,} respondents whose raw pred was out of (0, 1) "
              f"for either item")

    cal1 = _sigmoid(b0 + b1 * lg1[mask])
    cal2 = _sigmoid(b0 + b1 * lg2[mask])
    out = pd.DataFrame({
        "yearid_id": wide.loc[mask, "yearid_id"].to_numpy(),
        "year": wide.loc[mask, "year"].astype(int).to_numpy(),
        "pred_marhomo1_cal": cal1.astype("float32"),
        "pred_marhomo_cal":  cal2.astype("float32"),
    })

    out_path = OUT / "samesexframing_paired_individual.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(out):,} paired respondent predictions)")
    print(f"  mean marhomo1 cal = {out['pred_marhomo1_cal'].mean() * 100:.2f}%")
    print(f"  mean marhomo  cal = {out['pred_marhomo_cal'].mean() * 100:.2f}%")
    print(f"  mean diff         = {(out['pred_marhomo_cal'] - out['pred_marhomo1_cal']).mean() * 100:+.2f} pp")


if __name__ == "__main__":
    main()
