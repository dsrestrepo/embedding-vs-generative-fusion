#!/bin/bash
#SBATCH --job-name=gemma_zero_shot
#SBATCH --output=outputs/gemma_zero_shot_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=24:00:00

set -euo pipefail

# Load the Anaconda or Miniforge module
module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none

# Activate the Conda environment
source activate base_ml

# Models are pre-downloaded in the shared Ruche cache. Offline mode avoids
# gated-repository authentication checks during scheduled inference.
export HF_HOME=/gpfs/workdir/restrepoda/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Starting configured Gemma Zero-Shot Evaluation..."

args=(--config configs/paper_experiments.yaml)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    args+=(--case_index "$SLURM_ARRAY_TASK_ID")
fi

python scripts/eval_gemma_zero_shot.py "${args[@]}"

echo "Gemma Zero-Shot evaluation completed."
