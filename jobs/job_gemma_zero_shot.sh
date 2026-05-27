#!/bin/bash
#SBATCH --job-name=gemma_zero_shot
#SBATCH --output=outputs/gemma_zero_shot.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=24:00:00

# Load the Anaconda or Miniforge module
module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none

# Activate the Conda environment
source activate base_ml

echo "Starting configured Gemma Zero-Shot Evaluation..."

python scripts/eval_gemma_zero_shot.py --config configs/paper_experiments.yaml

echo "Gemma Zero-Shot evaluation completed."
