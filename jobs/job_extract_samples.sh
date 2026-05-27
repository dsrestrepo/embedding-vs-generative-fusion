#!/bin/bash
#SBATCH --job-name=extract_data_samples
#SBATCH --output=outputs/extract_data_samples.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu_short
#SBATCH --mem=32000
#SBATCH --time=01:00:00

# Load the Anaconda module
module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none

# Activate the Conda environment
source activate base_ml

# Move to the project root directory
# cd $SLURM_SUBMIT_DIR

# Create outputs directory if it doesn't exist
mkdir -p outputs
mkdir -p data_samples

# Run the python script
# We assume the job is submitted from the project root: sbatch jobs/job_extract_samples.sh
python scripts/extract_data_samples.py --dataset_root /gpfs/workdir/restrepoda/datasets --output_root data_samples
