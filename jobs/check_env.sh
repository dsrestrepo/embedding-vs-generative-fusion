#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --output=outputs/check_env.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu_short
#SBATCH --mem=4000

# Load the Anaconda module
module load anaconda3/2024.06/gcc-13.2.0

# Activate the Conda environment
source activate base_ml

#echo "Installing open_clip_torch==2.23.0 ..."
pip install open_clip_torch==2.23.0

echo "============================================"
echo "Python Configuration"
echo "============================================"
echo "Python Version:"
python --version
echo ""
echo "Python Location:"
which python
echo ""

echo "============================================"
echo "Installed Packages (pip freeze)"
echo "============================================"


pip freeze
pip freeze > requirements.txt
echo "Frozen requirements saved to requirements.txt"

echo ""
echo "============================================"
echo "Conda Environment Info"
echo "============================================"
conda info

# print head of pandas dataframe in /gpfs/workdir/restrepoda/datasets/coco-qa/labels.csv and the column names to verify the dataset is accessible and correctly formatted
#python -c "import pandas as pd; df = pd.read_csv('/gpfs/workdir/restrepoda/datasets/coco-qa/labels.csv'); print(df.head()); print(df.columns);"

# print df in the column path and path_preproc
#python -c "import pandas as pd; df = pd.read_csv('/gpfs/workdir/restrepoda/datasets/coco-qa/labels.csv'); print(df[['image_id', 'labels']].head());"

echo ""
echo "============================================"
echo "Check GPU Availability (if torch is installed)"
echo "============================================"
python -c "import torch; print('Torch available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())" 2>/dev/null || echo "Torch not installed or error checking CUDA"
