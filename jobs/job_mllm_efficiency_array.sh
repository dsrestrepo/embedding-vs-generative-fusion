#!/bin/bash
#SBATCH --job-name=mllm_efficiency
#SBATCH --output=outputs/mllm_efficiency/logs/%A_%a.out
# Run the full configured direct-MLLM profiling matrix sequentially so the
# large model checkpoint can be reused safely on a single GPU.
#SBATCH --array=0-7%1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=08:00:00

module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none
source activate base_ml
mkdir -p outputs/mllm_efficiency/logs
python scripts/benchmark_mllm_efficiency.py --config configs/paper_experiments.yaml --case_index "$SLURM_ARRAY_TASK_ID" --samples 20 --warmup 2 --max_new_tokens 16 --allow_download --output_dir outputs/mllm_efficiency
