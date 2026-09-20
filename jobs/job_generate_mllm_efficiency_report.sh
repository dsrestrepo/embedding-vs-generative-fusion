#!/bin/bash
#SBATCH --job-name=mllm_efficiency_report
#SBATCH --output=outputs/mllm_efficiency/report_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu_short
#SBATCH --mem=16000
#SBATCH --time=00:30:00

module load miniforge3/25.3.0-3/none-none
source activate base_ml
python scripts/generate_mllm_efficiency_report.py --efficiency_dir outputs/mllm_efficiency --zero_shot_dir outputs/gemma_zero_shot
