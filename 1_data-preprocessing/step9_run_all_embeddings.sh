#!/usr/bin/env bash
# Step 9: Generate embeddings for all three models.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python step9_generate_embeddings.py alpaca --model_name=maicomputer/alpaca-native
python step9_generate_embeddings.py gpt --model_name=EleutherAI/gpt-j-6B
python step9_generate_embeddings.py bert --model_name=roberta-large
