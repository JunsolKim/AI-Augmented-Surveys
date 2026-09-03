#!/usr/bin/env bash
#SBATCH --job-name=ft_s3_downstream
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

REPO="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/.." && pwd)"

# Runs stage 3 for ONE (variant × split) combination so multiple of these can
# be submitted in parallel via the pipeline launcher.

set -euo pipefail
cd "$REPO/5_finetuning-llm"

K=${K:-0}
N=${N:-200000}
LORA_TARGETS=${LORA_TARGETS:-all-linear}
LORA_R=${LORA_R:-16}
SPLIT=${SPLIT:-impute}              # single split per job

mkdir -p $REPO/logs
LOG=$REPO/logs/$(date +%Y-%m-%d_%H-%M-%S)_stage3_${LORA_TARGETS}_${SPLIT}_k${K}_n${N}.log

K="${K}" N="${N}" LORA_TARGETS="${LORA_TARGETS}" LORA_R="${LORA_R}" SPLITS="${SPLIT}" \
    bash step3_run_downstream.sh 2>&1 | tee "${LOG}"
