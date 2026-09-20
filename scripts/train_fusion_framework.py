import argparse
import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.framework.fusion import FusionTrainerFramework
from scripts.fusion_models import METHODS


def write_predictions(path, test_df, y_true, probabilities, gates):
    result = pd.DataFrame({"sample_id": test_df.index.to_numpy(), "y_true": y_true, "y_pred": probabilities.argmax(axis=1)})
    for i in range(probabilities.shape[1]):
        result[f"prob_{i}"] = probabilities[:, i]
    if gates is not None:
        result["gate_text"], result["gate_image"] = gates[:, 0], gates[:, 1]
    result.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run one frozen-embedding fusion baseline.")
    parser.add_argument("--embeddings_csv", required=True)
    parser.add_argument("--label_col", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output_dir", default="outputs/rebuttal_fusion")
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--fusion_method", choices=METHODS, required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multilabel", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    framework = FusionTrainerFramework(args.embeddings_csv, args.label_col, num_samples=args.num_samples,
                                       is_multilabel=args.multilabel, dataset_name=args.dataset)
    metrics, test_df, y_true, probabilities, gates, model = framework.run_method(
        args.fusion_method, random_state=args.seed, num_epochs=args.num_epochs,
        batch_size=args.batch_size, lr=args.lr, patience=args.patience,
    )
    metrics.update({"dataset": args.dataset, "backbone": args.backbone, "method": args.fusion_method, "seed": args.seed})
    stem = f"{args.dataset}_{args.backbone}_{args.fusion_method}"
    pd.DataFrame([metrics]).to_csv(os.path.join(args.output_dir, f"{stem}_results.csv"), index=False)
    write_predictions(os.path.join(args.output_dir, f"{stem}_predictions.csv"), test_df, y_true, probabilities, gates)
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{stem}_model.pt"))
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
