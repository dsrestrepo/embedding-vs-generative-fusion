#!/bin/bash
#SBATCH --job-name=download_datasets
#SBATCH --output=outputs/download_datasets.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_long       
#SBATCH --mem=32000                

# Load the Anaconda module
module load anaconda3/2024.06/gcc-13.2.0

# Activate the Conda environment
source activate base_ml

echo "Starting configured download job..."

python scripts/download_datasets.py --config configs/paper_experiments.yaml

echo "All requested downloads completed."
