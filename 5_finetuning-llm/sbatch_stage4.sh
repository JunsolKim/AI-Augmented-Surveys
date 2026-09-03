#!/usr/bin/env bash
#SBATCH --job-name=ft_s4_compare
#SBATCH --requeue
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

REPO="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/.." && pwd)"

set -euo pipefail
PY=python
cd "$REPO/5_finetuning-llm"

K=${K:-0}
N=${N:-200000}
LORA_R=${LORA_R:-16}
VARIANTS=${VARIANTS:-"all-linear,qv"}
SPLITS=${SPLITS:-"impute,partial,total"}

mkdir -p $REPO/logs
LOG=$REPO/logs/$(date +%Y-%m-%d_%H-%M-%S)_stage4_k${K}_n${N}.log

"${PY}" step4_compare.py \
    --k="${K}" --n="${N}" --lora_r="${LORA_R}" \
    --variants="${VARIANTS}" --splits="${SPLITS}" 2>&1 | tee "${LOG}"
