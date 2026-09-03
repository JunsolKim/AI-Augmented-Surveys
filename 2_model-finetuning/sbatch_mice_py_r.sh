#!/usr/bin/env bash
#SBATCH --job-name=micepyr
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#
# MICE (mice.impute.logreg emulation), parallelized over columns.
# Submit from 2_model-finetuning/ (after `mkdir -p logs`), one job per split/fold:
#   sbatch --job-name=micepyr_impute_k0 --export=ALL,SPLIT=impute,K=0 sbatch_mice_py_r.sh
# The downstream scripts read impute/partial for K=0..9 and mar/mnar for K=0.
set -euo pipefail

: "${SPLIT:?SPLIT must be exported (impute|partial|mar|mnar|module|panel)}"
: "${K:=0}"
: "${MAXIT:=15}"
: "${N_FEAT:=40}"
: "${M_DRAWS:=5}"
: "${N_CORES:=20}"

REPO="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/.." && pwd)"
PY=python
cd "$REPO/2_model-finetuning"

# Pin BLAS threads to 1 so joblib workers don't fight over CPU.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Predictor selection needs the global pairwise-correlation cache.
CORR="$REPO/data/cache/pairwise_corr_full.parquet"
if [ ! -s "$CORR" ]; then
  echo "[$(date)] $CORR missing - running precompute_pairwise_corr.py first"
  $PY precompute_pairwise_corr.py
fi

echo "[$(date)] Python MICE (logreg)  split=${SPLIT}  k=${K}  maxit=${MAXIT}  n_feat=${N_FEAT}  m_draws=${M_DRAWS}  n_cores=${N_CORES}"
$PY step1_mice_py_r.py run \
  --split_type="${SPLIT}" \
  --k="${K}" \
  --maxit="${MAXIT}" \
  --n_feat="${N_FEAT}" \
  --m_draws="${M_DRAWS}" \
  --n_cores="${N_CORES}"
echo "[$(date)] Done."
