# AI-Augmented Surveys — Replication Package

Replication code for:

> Kim, J., & Lee, B. **AI-Augmented Surveys: Leveraging Large Language Models
> for Opinion Prediction in Nationally Representative Surveys.**
> arXiv:2305.09620. <https://arxiv.org/abs/2305.09620>

## Directory layout

- **`1_data-preprocessing/`** — download and clean the GSS data and construct the model inputs (including LLM embeddings).
- **`2_model-finetuning/`** — train the prediction models and generate out-of-sample predictions.
- **`3_analysis-prediction/`** — aggregate the predictions into the estimates reported in the paper.
- **`4_figures-tables/`** — generate the figures and tables.
- **`5_finetuning-llm/`** — the LLM fine-tuning experiment (Figure A22).
- **`data/`** — input data and intermediate files (downloaded separately; see Data).
- **`output/`** — generated figures and tables.

## Setup

```bash
./setup.sh                          # creates the conda environment (Python + R)
conda activate ai-augmented-surveys
```

## Data

The data is on OSF: <https://osf.io/qx5ht>. Download `ai-augmented-surveys-data.tar.gz`
and extract it in the repository root, which populates `data/`:

```bash
tar xzf ai-augmented-surveys-data.tar.gz
```

Trained model weights are on a separate OSF project, <https://osf.io/ju5xw>,
split into 15 parts (~52 GB). They are
only needed to regenerate the predictions without retraining:

```bash
cd data && cat ai-augmented-surveys-weights.tgz.part-* | gunzip -c | tar xf -
```

## Reproducing the figures and tables

All figures and tables are produced from the shipped `data/fig_table_gen/` on a
CPU in a few minutes. Snakemake is not needed for this step:

```bash
./run_figures.sh        # writes output/
```

## Rebuilding the intermediates or the full pipeline

Stages 1–3 are orchestrated with Snakemake (installed by `environment.yml`).
Each stage has its own `Snakefile`; the root `Snakefile` covers stages 3 and 4.

```bash
# Stage 1: raw GSS -> analysis data, demographics, prompts, LLM embeddings (GPU)
cd 1_data-preprocessing && snakemake --cores 8

# Stage 2: train + predict (GPU, SLURM), then aggregate to variable-year means and run MICE
cd 2_model-finetuning && mkdir -p logs
snakemake --profile slurm-profile main   # edit slurm-profile/config.yaml and env_setup.sh for your cluster
bash run_long_to_varyear.sh
sbatch --export=ALL,SPLIT=impute,K=0 sbatch_mice_py_r.sh   # impute/partial for K=0..9, mar/mnar for K=0

# Stage 3: predictions -> data/fig_table_gen/
cd .. && snakemake --cores 8 analysis

# Stage 4 (same as run_figures.sh)
snakemake --cores 8 figures
```

`run_all.sh` runs these stages in order. Notes:

- Stages 1, 2, and 5 need a CUDA GPU; stage 2 assumes a SLURM cluster.
- Set `export TF_USE_LEGACY_KERAS=1` before stage 2 (`env_setup.sh` does this).
  If TensorFlow fails with `libdevice not found`, point XLA at a CUDA toolkit:
  `export XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda`.
- `step3_long_to_wide.py` needs roughly 500 GB of memory.
- The Roper Center iPoll data used for Figures 4 and 5 and Table A12 is
  not included. 
  It is available from the authors upon request, for replication purposes only.
  The scripts that read it (`3_analysis-prediction/roper_*.py`,
  `prep_roper_question_meta.py`, `prep_roper_by_existence.py`,
  `prep_counterfactual_roper.py`) are included for reference and are not part
  of the Snakemake DAG; their aggregated outputs ship in `data/fig_table_gen/`.
