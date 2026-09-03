#!/usr/bin/env bash
# Create the conda environment (Python + R) for this replication package.
set -euo pipefail
cd "$(dirname "$0")"

conda env create -f environment.yml

# estimatr (HC1 robust SEs in the R figure scripts) is installed after the env
# exists: conda-forge has no build for every platform, so fall back to CRAN.
echo
echo "Installing estimatr..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ai-augmented-surveys
if conda install -y -c conda-forge r-estimatr 2>/dev/null; then
    echo "  installed from conda-forge"
else
    echo "  no conda-forge build for this platform; building from CRAN"
    echo "  (needs a C++ toolchain; on macOS install Xcode command line tools)"
    Rscript -e 'if (!requireNamespace("estimatr", quietly = TRUE)) install.packages("estimatr", repos = "https://cloud.r-project.org")'
fi
Rscript -e 'if (!requireNamespace("estimatr", quietly = TRUE)) { cat("ERROR: estimatr unavailable\n"); quit(status = 1) }'

echo
echo "Environment 'ai-augmented-surveys' created (Python + R)."
echo "Activate it with:  conda activate ai-augmented-surveys"
echo "For the model-training stage also:  export TF_USE_LEGACY_KERAS=1"
