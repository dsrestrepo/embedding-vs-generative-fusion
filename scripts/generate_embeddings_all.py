import subprocess
import os
import sys
import argparse
from typing import Any, Dict, List

import yaml

# Configuration
# The root directory where datasets are stored
DATASET_ROOT = '/gpfs/workdir/restrepoda/datasets'
# DATASET_ROOT = './datasets' # Uncomment for local testing

# The root directory where outputs will be saved
OUTPUT_ROOT = '/gpfs/workdir/restrepoda/Embeddings_vlm'

# Training parameters
BATCH_SIZE = 256
NPROC_PER_NODE = 2

# Dataset configurations
# 'path': relative path from DATASET_ROOT
# 'image_col': column name for images
# 'text_col': column name for text
DATASET_CONFIGS = {
    'daquar': {'path': 'daquar', 'image_col': 'image_id', 'text_col': 'question', 'image_dir': 'images'},
    'coco-qa': {'path': 'coco-qa', 'image_col': 'image_id', 'text_col': 'questions', 'image_dir': 'images'},
    'fakeddit': {'path': 'fakeddit', 'image_col': 'id', 'text_col': 'text', 'image_dir': 'images'},
    'Recipes5k': {'path': 'Recipes5k', 'image_col': 'image', 'text_col': 'ingredients', 'image_dir': 'images'},
    'brset': {'path': 'BRSET/brset', 'image_col': 'image_id', 'text_col': 'text', 'image_dir': 'images'},
    'ham10000': {'path': 'HAM10000', 'image_col': 'image_id', 'text_col': 'text', 'image_dir': 'images'},
    'mimic': {'path': 'MIMIC/mimic', 'image_col': 'path_preproc', 'text_col': 'text', 'image_dir': '.'},
    'mbrset': {'path': 'mBRSET/mbrset', 'image_col': 'file', 'text_col': 'text', 'image_dir': 'images'},
}

# List of datasets to process (add/remove as needed)
DATASETS_TO_RUN = [
    'Recipes5k',    # ✅
    'daquar',       # ✅
    'coco-qa',      # ✅
    'brset',        # ✅
    'ham10000',     # ✅
    'mimic',        # ✅
    'mbrset',       # ✅ 
    'fakeddit',     # ✅
]

# List of models to use
MODELS_TO_RUN = ['CLIP', 'MedSigLip', 'SigLip', 'biomedclip'] #['CLIP', 'MedSigLip', 'SigLip', 'biomedclip', 'Unimed']

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def enabled_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in items if item.get("enabled", False)]


def run_command(command):
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")

def run_embeddings(datasets, models, dataset_root, output_root, batch_size, nproc_per_node):
    # Ensure subprocess is run from the project root where src/ matches
    if not os.path.exists('src/vlm_embeddings.py'):
        print("Error: src/vlm_embeddings.py not found. Please run this script from the project root.")
        return

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for dataset_name, config in datasets:
        dataset_path = os.path.join(dataset_root, config['path'])
        
        # Check availability (warn only)
        if not os.path.exists(dataset_path):
             print(f"Warning: Dataset path {dataset_path} does not exist (or is not accessible). Skipping {dataset_name}?")
             pass

        for model in models:
            output_dir = os.path.join(output_root, dataset_name)
            output_file = f'embeddings_{model.lower()}.csv'
            
            cmd = [
                "torchrun",
                f"--nproc_per_node={nproc_per_node}",
                "src/vlm_embeddings.py",
                "--classifier",
                model,
                "--batch_size",
                str(batch_size),
                "--dataset_path",
                dataset_path,
                "--image_col",
                config['image_col'],
                "--text_col",
                config['text_col'],
                "--output_dir",
                output_dir,
                "--output_file",
                output_file,
                "--image_dir",
                config.get('image_dir', 'images'),
            ]

            print(f"\n--- Processing {dataset_name} with {model} ---")
            run_command(cmd)


def run_configured_embeddings(config_path):
    config = load_config(config_path)
    paths = config.get("paths", {})
    embeddings_cfg = config.get("embeddings", {})

    datasets = [
        (dataset["name"], dataset)
        for dataset in enabled_items(embeddings_cfg.get("datasets", []))
    ]
    models = [model["name"] for model in enabled_items(embeddings_cfg.get("models", []))]

    if not datasets:
        print("No enabled datasets found in embeddings config.")
        return
    if not models:
        print("No enabled models found in embeddings config.")
        return

    run_embeddings(
        datasets=datasets,
        models=models,
        dataset_root=paths.get("data_dir", "datasets"),
        output_root=paths.get("embeddings_dir", "Embeddings_vlm"),
        batch_size=embeddings_cfg.get("batch_size", 256),
        nproc_per_node=embeddings_cfg.get("nproc_per_node", 1),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate VLM embeddings for configured datasets.")
    parser.add_argument("--config", type=str, help="Run enabled embeddings entries from a YAML config")
    args = parser.parse_args()

    if args.config:
        run_configured_embeddings(args.config)
        return

    datasets = []
    for dataset_name in DATASETS_TO_RUN:
        if dataset_name not in DATASET_CONFIGS:
            print(f"Warning: Configuration for {dataset_name} not found. Skipping.")
            continue
        datasets.append((dataset_name, DATASET_CONFIGS[dataset_name]))

    run_embeddings(
        datasets=datasets,
        models=MODELS_TO_RUN,
        dataset_root=DATASET_ROOT,
        output_root=OUTPUT_ROOT,
        batch_size=BATCH_SIZE,
        nproc_per_node=NPROC_PER_NODE,
    )

if __name__ == "__main__":
    main()
