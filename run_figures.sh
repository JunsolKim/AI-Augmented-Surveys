#!/usr/bin/env bash
# Reproduce all figures and tables from the shipped data (CPU-only).
# Requires: `data/` populated (see README > Data) and the conda env activated.
set -euo pipefail
cd "$(dirname "$0")/4_figures-tables"

for f in figure*.py table*.py; do
    echo ">> $f"; python "$f"
done
for f in figure*.R; do
    echo ">> $f"; Rscript "$f"
done

# Figure A22 lives in the fine-tuning stage.
echo ">> figure_ft_effect.R"
Rscript ../5_finetuning-llm/figure_ft_effect.R

echo "Done. Figures and tables are in ../output/"
