#!/usr/bin/env bash
#SBATCH --job-name=ft_s2_extract
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

REPO="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/.." && pwd)"

set -euo pipefail
PY=python
cd "$REPO/5_finetuning-llm"

K=${K:-0}
N=${N:-200000}
LORA_TARGETS=${LORA_TARGETS:-all-linear}
LORA_R=${LORA_R:-16}

mkdir -p $REPO/logs
LOG=$REPO/logs/$(date +%Y-%m-%d_%H-%M-%S)_stage2_${LORA_TARGETS}_k${K}_n${N}.log

"${PY}" step2_extract_embeddings.py \
    --k="${K}" --n="${N}" \
    --lora_targets="${LORA_TARGETS}" --lora_r="${LORA_R}" 2>&1 | tee "${LOG}"
