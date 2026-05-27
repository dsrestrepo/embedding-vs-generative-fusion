# scripts/train_fusion_framework.py

import os
import argparse
import pandas as pd
import sys
from typing import Any, Dict, List

import yaml

# Framework usage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.framework.fusion import FusionTrainerFramework

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def enabled_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in items if item.get("enabled", False)]


def run_training(args):
    missing = [
        name
        for name in ("embeddings_csv", "label_col")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(f"Missing required arguments for single run: {', '.join(missing)}")

    os.makedirs(args.output_dir, exist_ok=True)

    fusion_framework = FusionTrainerFramework(args.embeddings_csv, args.label_col, num_samples=args.num_samples, is_multilabel=args.multilabel, dataset_name=args.dataset)

    if args.fusion_method == "linear":
        metrics = fusion_framework.run_early_fusion_baseline()
    elif args.fusion_method == "mlp_early":
        metrics = fusion_framework.run_mlp_fusion(fusion_type="early", num_epochs=args.num_epochs)
    elif args.fusion_method == "mlp_late":
        metrics = fusion_framework.run_mlp_fusion(fusion_type="late", num_epochs=args.num_epochs)

    print("\n========= METRICS =========")
    print(f"Dataset: {args.dataset} | Method: {args.fusion_method}")
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    if 'auc' in metrics:
        print(f"AUC: {metrics['auc']:.4f}")
    print(f"F1 Score: {metrics.get('f1_score', 0):.4f}")

    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.backbone}_{args.fusion_method}_results.csv")
    metrics['dataset'] = args.dataset
    metrics['backbone'] = args.backbone
    pd.DataFrame([metrics]).to_csv(output_file, index=False)
    print(f"Metrics saved to {output_file}")


def run_configured_training(config_path: str):
    config = load_config(config_path)
    paths = config.get("paths", {})
    embeddings_dir = paths.get("embeddings_dir", "Embeddings_vlm")

    fusion_cfg = config.get("framework_fusion", {})
    output_dir = fusion_cfg.get("output_dir", "outputs/framework_fusion")
    methods = fusion_cfg.get("methods", [])
    datasets = enabled_items(fusion_cfg.get("datasets", []))

    if not datasets:
        print("No enabled datasets found in framework_fusion config.")
        return

    for dataset in datasets:
        for backbone in dataset.get("backbones", []):
            for method in methods:
                embeddings_csv = os.path.join(
                    embeddings_dir,
                    dataset["embeddings_subdir"],
                    f"embeddings_{backbone}.csv",
                )
                args = argparse.Namespace(
                    embeddings_csv=embeddings_csv,
                    label_col=(
                        [dataset["label_col"]]
                        if isinstance(dataset["label_col"], str)
                        else dataset["label_col"]
                    ),
                    dataset=dataset["name"],
                    backbone=backbone,
                    output_dir=output_dir,
                    num_samples=None,
                    fusion_method=method,
                    num_epochs=fusion_cfg.get("num_epochs", 100),
                    multilabel=dataset.get("multilabel", False),
                )
                print(
                    f"--- Running Fusion ({method}) on {args.dataset} [Backbone: {backbone}] ---"
                )
                run_training(args)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Embeddings using Data Fusion Framework")
    parser.add_argument("--config", type=str, help="Run enabled framework_fusion entries from a YAML config")
    parser.add_argument("--embeddings_csv", type=str, help="Path to pre-extracted embeddings")
    parser.add_argument("--label_col", type=str, nargs='+', help="Column name(s) with ground truth label")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset name for logging")
    parser.add_argument("--backbone", type=str, default="clip", help="Backbone model used for the embeddings")
    parser.add_argument("--output_dir", type=str, default="outputs/framework_fusion/", help="Output folder")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of rows to subsample for faster training")
    parser.add_argument("--fusion_method", type=str, choices=["linear", "mlp_early", "mlp_late"], default="linear", help="Which model to run")
    parser.add_argument("--num_epochs", type=int, default=100, help="Epochs if using MLP")
    parser.add_argument("--multilabel", action="store_true", help="Set flag if classification is multilabel (e.g. MIMIC, Daquar)")
    
    args = parser.parse_args()

    if args.config:
        run_configured_training(args.config)
    else:
        run_training(args)

if __name__ == "__main__":
    main()
