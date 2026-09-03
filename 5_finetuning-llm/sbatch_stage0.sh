#!/usr/bin/env bash
#SBATCH --job-name=ft_s0_pretrain
#SBATCH --time=02:00:00
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
EPOCHS=${EPOCHS:-2}

mkdir -p $REPO/logs
LOG=$REPO/logs/$(date +%Y-%m-%d_%H-%M-%S)_stage0_k${K}.log

"${PY}" step0_pretrain_head.py --k="${K}" --epochs="${EPOCHS}" 2>&1 | tee "${LOG}"
