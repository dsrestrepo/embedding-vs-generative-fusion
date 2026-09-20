"""Paired bootstrap confidence intervals from saved held-out predictions."""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


METRICS = ("accuracy", "macro_f1", "auroc")


def score(frame, indices=None):
    if indices is not None:
        frame = frame.iloc[indices]
    y = frame.y_true.to_numpy(); pred = frame.y_pred.to_numpy()
    probs = frame[[c for c in frame if c.startswith("prob_")]].to_numpy()
    result = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}
    try:
        result["auroc"] = roc_auc_score(y, probs[:, 1]) if probs.shape[1] == 2 else roc_auc_score(y, probs, multi_class="ovr", average="macro")
    except ValueError:
        result["auroc"] = np.nan
    return result


def bootstrap(frame, rng, samples):
    n = len(frame); values = {metric: np.empty(samples) for metric in METRICS}
    for i in range(samples):
        result = score(frame, rng.integers(0, n, n))
        for metric in METRICS: values[metric][i] = result[metric]
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/rebuttal_fusion")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    files = glob.glob(os.path.join(args.results_dir, "*_predictions.csv"))
    if not files: raise FileNotFoundError("No prediction files found.")
    rows, frames = [], {}
    for path in files:
        stem = os.path.basename(path).removesuffix("_predictions.csv")
        dataset, backbone, method = stem.split("_", 2)
        frame = pd.read_csv(path); frames[(dataset, backbone, method)] = frame
        values = bootstrap(frame, np.random.default_rng(args.seed), args.samples)
        for metric, draws in values.items():
            point = score(frame)[metric]
            rows.append({"dataset": dataset, "backbone": backbone, "method": method, "metric": metric,
                         "estimate": point, "ci_low": np.nanquantile(draws, .025), "ci_high": np.nanquantile(draws, .975),
                         "bootstrap_samples": args.samples})
    pd.DataFrame(rows).to_csv(os.path.join(args.results_dir, "bootstrap_summary.csv"), index=False)
    comparisons = (
        ("gated_attention", "mlp_late"),
        ("mcr_late", "mlp_late"),
        ("gated_attention", "mlp_early"),
        ("i2moe", "mlp_early"),
        ("i2moe", "mlp_late"),
    )
    paired = []
    for (dataset, backbone, method), left in frames.items():
        for a, b in comparisons:
            if method != a or (dataset, backbone, b) not in frames: continue
            merged = left.merge(frames[(dataset, backbone, b)], on="sample_id", suffixes=("_a", "_b"), validate="one_to_one")
            # y_true must be identical because the split is shared across all methods.
            if not np.array_equal(merged.y_true_a, merged.y_true_b): raise RuntimeError("Prediction files have mismatched test labels.")
            for metric in METRICS:
                draws = []
                rng = np.random.default_rng(args.seed); n = len(merged)
                for _ in range(args.samples):
                    idx = rng.integers(0, n, n)
                    a_frame = merged[[c for c in merged if c.endswith("_a")]].rename(columns=lambda c: c.removesuffix("_a"))
                    b_frame = merged[[c for c in merged if c.endswith("_b")]].rename(columns=lambda c: c.removesuffix("_b"))
                    draws.append(score(a_frame, idx)[metric] - score(b_frame, idx)[metric])
                a_frame = merged[[c for c in merged if c.endswith("_a")]].rename(columns=lambda c: c.removesuffix("_a"))
                b_frame = merged[[c for c in merged if c.endswith("_b")]].rename(columns=lambda c: c.removesuffix("_b"))
                paired.append({"dataset": dataset, "backbone": backbone, "comparison": f"{a} - {b}", "metric": metric,
                               "difference": score(a_frame)[metric] - score(b_frame)[metric],
                               "ci_low": np.nanquantile(draws, .025), "ci_high": np.nanquantile(draws, .975),
                               "bootstrap_samples": args.samples})
    pd.DataFrame(paired).to_csv(os.path.join(args.results_dir, "bootstrap_pairwise.csv"), index=False)


if __name__ == "__main__": main()
