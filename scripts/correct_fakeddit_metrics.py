"""Reinterpret predictions produced by the legacy inverted Fakeddit prompt."""

import argparse
import os
import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def parse_legacy_response(value):
    """Map legacy prompt semantics (0=Real, 1=Fake) to dataset labels."""
    response = str(value).strip().lower()
    numeric = re.match(r"^\s*([01])\b", response)
    if numeric:
        return 1 if numeric.group(1) == "0" else 0
    if re.search(r"\breal\b", response):
        return 1
    if re.search(r"\bfake\b", response):
        return 0
    raise ValueError(f"Cannot parse saved Fakeddit response: {value!r}")


def correct_file(path, output_dir):
    frame = pd.read_csv(path)
    parsed = frame["gemma_prediction"].map(parse_legacy_response)
    frame["gemma_parsed_prediction"] = parsed
    frame.to_csv(path, index=False)

    y_true = frame["2_way_label"].astype(int)
    accuracy = accuracy_score(y_true, parsed)
    macro_f1 = f1_score(y_true, parsed, average="macro")
    model = os.path.basename(path).removeprefix("fakeddit_").removesuffix("_predictions.csv")
    metrics_path = os.path.join(output_dir, f"fakeddit_{model}_Zero-Shot_results.csv")
    pd.DataFrame([{
        "model": "Zero-Shot LLM",
        "dataset": "fakeddit",
        "backbone": model,
        "fusion_method": "Zero-Shot",
        "accuracy": accuracy,
        "f1_score": macro_f1,
        "auc": 0.0,
    }]).to_csv(metrics_path, index=False)
    print(f"{model}: accuracy={accuracy:.6f}, macro_f1={macro_f1:.6f}")
    print(f"Updated {path}")
    print(f"Updated {metrics_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", default="outputs/gemma_zero_shot")
    args = parser.parse_args()
    paths = [
        os.path.join(args.predictions_dir, f"fakeddit_{model}_predictions.csv")
        for model in ("gemma-3-27b-it", "gemma-3-4b-it")
    ]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        correct_file(path, args.predictions_dir)

if __name__ == "__main__":
    main()
