# Sourced by every Snakemake rule in this stage before running Python.
# Adapt the site-specific block for your cluster; the exports below it are
# required everywhere. Not executable on purpose: it is sourced, never run.

# --- site-specific ----------------------------------------------------------
# Load a CUDA toolkit here if your cluster uses environment modules, e.g.
#   module load cuda/12.2
# XLA needs libdevice from a CUDA toolkit; pip-installed TensorFlow does not
# ship it. If GPU ops fail with "libdevice not found" / "JIT compilation
# failed", point XLA at the toolkit:
#   export XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda

# --- required everywhere ----------------------------------------------------
# tensorflow_recommenders needs Keras 2 semantics.
export TF_USE_LEGACY_KERAS=1

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    . "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ai-augmented-surveys 2>/dev/null || true
fi

# Fail loudly rather than silently running the wrong interpreter.
python -c "import tensorflow, tensorflow_recommenders" 2>/dev/null || {
    echo "env_setup.sh: expected env not active (python=$(command -v python))" >&2
    return 1 2>/dev/null || exit 1
}
