#!/usr/bin/env bash
# Step 3: Download GSS cumulative cross-sectional Stata file from NORC.
#
# Input:  None (downloads from NORC server)
# Output: {DATA_DIR}/gss7221_r2.dta

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading GSS cross-sectional data to $DATA_DIR ..."
curl -L https://gss.norc.org/documents/stata/GSS_stata.zip -o GSS_stata.zip

echo "Extracting ..."
unzip -o GSS_stata.zip

echo "Done."
