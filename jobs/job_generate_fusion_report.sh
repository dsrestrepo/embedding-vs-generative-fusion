#!/bin/bash
#SBATCH --job-name=dfdm_fusion_report
#SBATCH --output=outputs/rebuttal_fusion/fusion_report_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=02:00:00

module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none
source activate base_ml
python scripts/generate_fusion_report.py --results_dir outputs/rebuttal_fusion
