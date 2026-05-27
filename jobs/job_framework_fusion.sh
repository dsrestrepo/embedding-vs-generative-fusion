#!/bin/bash
#SBATCH --job-name=framework_fusion_train
#SBATCH --output=outputs/framework_fusion_train.out
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

echo "Starting configured Fusion Framework Training..."

python scripts/train_fusion_framework.py --config configs/paper_experiments.yaml

echo "Framework Fusion training completed."
