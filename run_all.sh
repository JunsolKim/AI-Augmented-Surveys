#!/usr/bin/env bash
# Full pipeline: preprocessing -> training -> prediction -> analysis -> figures.
# Requires a CUDA GPU and a SLURM cluster. Run from the repo root with the conda
# env activated.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export TF_USE_LEGACY_KERAS=1

# 1. Preprocessing: GSS download -> analysis data -> demographics, prompts, embeddings.
cd "$ROOT/1_data-preprocessing"
snakemake --cores 8

# 2. Model training + prediction (DCN + MF), then per-var-year aggregation and MICE.
cd "$ROOT/2_model-finetuning"
mkdir -p logs
snakemake --profile slurm-profile main
bash run_long_to_varyear.sh
for SPLIT in impute partial; do
    for K in $(seq 0 9); do
        sbatch --wait --job-name=micepyr_${SPLIT}_k${K} --export=ALL,SPLIT=$SPLIT,K=$K sbatch_mice_py_r.sh &
    done
done
for SPLIT in mar mnar; do
    sbatch --wait --job-name=micepyr_${SPLIT}_k0 --export=ALL,SPLIT=$SPLIT,K=0 sbatch_mice_py_r.sh &
done
wait

# 3-4. Aggregate predictions into per-figure/table estimates, then render.
cd "$ROOT"
snakemake --cores 8 analysis
snakemake --cores 8 figures
