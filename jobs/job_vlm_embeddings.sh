#!/bin/bash
#SBATCH --job-name=vlm_embeddings
#SBATCH --output=outputs/vlm_embeddings.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=gpua100
#SBATCH --mem=120000
#SBATCH --time=24:00:00

# Load the Anaconda module
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/12.2.1/gcc-11.2.0

# Activate the Conda environment
source activate base_ml

# Move to the project root directory (assuming submission from there, but good to be explicit if fixed)
# cd $SLURM_SUBMIT_DIR

# Run configured embeddings
# We assume the job is submitted from the project root: sbatch jobs/job_vlm_embeddings.sh
python scripts/generate_embeddings_all.py --config configs/paper_experiments.yaml
