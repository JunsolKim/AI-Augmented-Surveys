#!/usr/bin/env bash

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Train and predict the DCN on the fine-tuned embedding pickle for the given splits.
#
# Usage:
#   LORA_TARGETS=all-linear  N=200000  bash step3_run_downstream.sh
#   LORA_TARGETS=qv          N=200000  bash step3_run_downstream.sh
#
# Optional overrides:
#   SPLITS="impute partial total"   # space-separated list
#   K=0
#   LORA_R=16
#
# The pickle MUST already exist at:
#   data/weights/alpaca-ft-intersection-k${K}-${LORA_TARGETS}-r${LORA_R}-n${N}.pkl

set -euo pipefail

K=${K:-0}
N=${N:-200000}
LORA_TARGETS=${LORA_TARGETS:-all-linear}
LORA_R=${LORA_R:-16}
SPLITS=${SPLITS:-"impute partial total"}

MODEL_NAME="alpaca-ft-intersection-k${K}-${LORA_TARGETS}-r${LORA_R}-n${N}"
REPL_DIR="$REPO"
PKL_PATH="${REPL_DIR}/data/weights/${MODEL_NAME}.pkl"

if [[ ! -f "${PKL_PATH}" ]]; then
    echo "[stage3] ERROR: embedding pickle not found at ${PKL_PATH}"
    echo "[stage3]   run step2_extract_embeddings.py first."
    exit 1
fi
echo "[stage3] model_name = ${MODEL_NAME}"
echo "[stage3] splits     = ${SPLITS}"

PY=python
cd "${REPL_DIR}/2_model-finetuning"

for ST in ${SPLITS}; do
    echo
    echo "============================================================"
    echo "[stage3] split_type=${ST}  k=${K}"
    echo "============================================================"
    "${PY}" step1_train.py    dcn      --model_name="${MODEL_NAME}" --split_type="${ST}" --k="${K}"
    "${PY}" step2_predict.py  dcn_fold --model_name="${MODEL_NAME}" --split_type="${ST}" --k="${K}"
done

echo
echo "[stage3] DONE. Predictions saved to ${REPL_DIR}/data/predictions/"
