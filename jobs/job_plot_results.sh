#!/bin/bash
#SBATCH --job-name=plot_results
#SBATCH --output=outputs/plot_results.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00

# Load the Anaconda or Miniforge module
module load miniforge3/25.3.0-3/none-none

# Activate the Conda environment
source activate base_ml

echo "Ensuring dependencies are installed..."
pip install tabulate

echo "Starting plot generation..."

python scripts/plot_framework_results.py

echo "Plotting completed. Check the outputs/framework_fusion/plots/ directory."