#!/bin/bash
#SBATCH --job-name=mllm_bootstrap
#SBATCH --output=outputs/mllm_bootstrap_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00

set -euo pipefail

module load miniforge3/25.3.0-3/none-none
source activate base_ml

python scripts/bootstrap_mllm_metrics.py \
    --predictions_dir outputs/gemma_zero_shot \
    --output_dir outputs/gemma_zero_shot \
    --samples 2000 \
    --expected_cases 8
