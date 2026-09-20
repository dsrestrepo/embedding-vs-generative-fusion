#!/bin/bash
#SBATCH --job-name=dfdm_rebuttal
#SBATCH --output=outputs/rebuttal_fusion/logs/%A_%a.out
#SBATCH --array=0-41%6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=08:00:00

module load miniforge3/25.3.0-3/none-none
module load cuda/12.2.2/none-none
source activate base_ml

datasets=(Recipes5k Recipes5k fakeddit fakeddit mbrset mbrset)
backbones=(clip siglip clip siglip biomedclip medsiglip)
labels=(class class 2_way_label 2_way_label DR_2 DR_2)
methods=(image_only text_only linear mlp_early mlp_late gated_attention mcr_late)

pair=$((SLURM_ARRAY_TASK_ID / 7)); method_index=$((SLURM_ARRAY_TASK_ID % 7))
dataset=${datasets[$pair]}; backbone=${backbones[$pair]}; label=${labels[$pair]}; method=${methods[$method_index]}
case "$dataset" in Recipes5k) subdir=Recipes5k ;; fakeddit) subdir=fakeddit ;; mbrset) subdir=mbrset ;; esac
mkdir -p outputs/rebuttal_fusion/logs
python scripts/train_fusion_framework.py \
  --embeddings_csv "/gpfs/workdir/restrepoda/Embeddings_vlm/$subdir/embeddings_${backbone}.csv" \
  --label_col "$label" --dataset "$dataset" --backbone "$backbone" --fusion_method "$method" \
  --output_dir outputs/rebuttal_fusion --num_epochs 100 --patience 5 --batch_size 4096 --seed 42
