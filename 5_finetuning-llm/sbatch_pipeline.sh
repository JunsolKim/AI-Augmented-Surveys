#!/usr/bin/env bash
# Fine-tuning pipeline launcher with Slurm dependency chaining.
#
# Dependency DAG:
#
#   stage0 (1 job, shared)
#     ├─→ stage1_all-linear ──→ stage2_all-linear ──┬─→ stage3_all-linear_impute  ─┐
#     │                                              ├─→ stage3_all-linear_partial ─┤
#     │                                              └─→ stage3_all-linear_total   ─┤
#     │                                                                              ├─→ stage4
#     └─→ stage1_qv         ──→ stage2_qv         ──┬─→ stage3_qv_impute          ─┤
#                                                    ├─→ stage3_qv_partial         ─┤
#                                                    └─→ stage3_qv_total           ─┘
#
# All siblings in a column run concurrently if the cluster has GPUs free.
# Slurm holds the rest in pending state until their dependencies satisfy.
#
# Usage:
#   bash sbatch_pipeline.sh                          # both variants, all splits
#   VARIANTS="all-linear" bash sbatch_pipeline.sh    # only one variant
#   SPLITS="impute total" bash sbatch_pipeline.sh    # only two splits

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

K=${K:-0}
N=${N:-200000}
LORA_R=${LORA_R:-16}
VARIANTS=${VARIANTS:-"all-linear qv"}
SPLITS=${SPLITS:-"impute partial total"}

echo "[pipeline] K=${K}  N=${N}  LORA_R=${LORA_R}"
echo "[pipeline] VARIANTS=${VARIANTS}  SPLITS=${SPLITS}"
echo

# -------- stage 0 (shared) --------
S0=$(sbatch --parsable --export=ALL,K="${K}" sbatch_stage0.sh)
echo "stage0:                                  ${S0}"

# -------- stages 1/2 (parallel per variant) + stage 3 (parallel per variant × split) --------
declare -a STAGE3_JOBS=()

for V in ${VARIANTS}; do
    echo
    echo "--- variant=${V} ---"
    EXPORT_BC="ALL,K=${K},N=${N},LORA_TARGETS=${V},LORA_R=${LORA_R}"

    S1=$(sbatch --parsable --dependency=afterok:${S0}  --export="${EXPORT_BC}" sbatch_stage1.sh)
    S2=$(sbatch --parsable --dependency=afterok:${S1}  --export="${EXPORT_BC}" sbatch_stage2.sh)
    echo "  stage1_${V}:                          ${S1}"
    echo "  stage2_${V}:                          ${S2}"

    for ST in ${SPLITS}; do
        EXPORT_D="${EXPORT_BC},SPLIT=${ST}"
        S3=$(sbatch --parsable --dependency=afterok:${S2} --export="${EXPORT_D}" sbatch_stage3.sh)
        echo "  stage3_${V}_${ST}:               ${S3}"
        STAGE3_JOBS+=("${S3}")
    done
done

# -------- stage 4 (one job, depends on all stage3 jobs) --------
VARIANTS_CSV=$(echo "${VARIANTS}" | tr ' ' ',')
SPLITS_CSV=$(echo "${SPLITS}" | tr ' ' ',')
DEP_LIST=$(IFS=: ; echo "${STAGE3_JOBS[*]}")

EXPORT_S4="ALL,K=${K},N=${N},LORA_R=${LORA_R},VARIANTS=${VARIANTS_CSV},SPLITS=${SPLITS_CSV}"
S4=$(sbatch --parsable --dependency=afterok:${DEP_LIST} --export="${EXPORT_S4}" sbatch_stage4.sh)
echo
echo "stage4 (compare):                        ${S4}"

echo
echo "[pipeline] watch with:"
echo "    squeue -u \$USER --states=PD,R --format='%.10i %.30j %.10T %.10M %R' | sort"
echo "[pipeline] results will land at:"
echo "    5_finetuning-llm/logs/stage4_compare_k${K}_n${N}.csv"
echo "    5_finetuning-llm/logs/stage4_delta_auc_k${K}_n${N}.pdf"
