#!/usr/bin/env bash
# Submit step5_long_to_varyear jobs, one per model (array 0-3).
# Each job loops over the split types; step5 skips those with no fold-0 long file.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

sbatch --wait <<'EOF2'
#!/bin/bash
#SBATCH --job-name=varyear
#SBATCH --mem=200G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/varyear_%a_%j.out
#SBATCH --error=logs/varyear_%a_%j.err

cd "$SLURM_SUBMIT_DIR"

# 0: alpaca, 1: mf, 2: roberta, 3: gpt-j
case $SLURM_ARRAY_TASK_ID in
    0) MODEL="maicomputer/alpaca-native" ;;
    1) MODEL="mf" ;;
    2) MODEL="roberta-large" ;;
    3) MODEL="EleutherAI/gpt-j-6B" ;;
esac

for SPLIT in impute partial total module; do
    echo "=== $MODEL / $SPLIT ==="
    python step5_long_to_varyear.py run \
        --model_name="$MODEL" \
        --split_type="$SPLIT"
done
EOF2

echo "varyear job array (4 models) finished"
