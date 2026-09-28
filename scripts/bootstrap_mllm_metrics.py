"""Bootstrap direct-MLLM accuracy and macro-F1 from saved predictions."""

import argparse
import glob
import os

import numpy as np
import pandas as pd


LABEL_COLUMNS = {
    "Recipes5k": "class",
    "fakeddit": "2_way_label",
    "mbrset": "DR_2",
}

MODEL_LABELS = {
    "gemma-3-27b-it": "Gemma-3-27B-IT",
    "gemma-3-4b-it": "Gemma-3-4B-IT",
    "medgemma-27b-it": "MedGemma-27B-IT",
    "medgemma-4b-it": "MedGemma-4B-IT",
}


def identify_case(path):
    stem = os.path.basename(path).removesuffix("_predictions.csv")
    for marker in ("_medgemma-", "_gemma-"):
        if marker in stem:
            dataset, suffix = stem.rsplit(marker, 1)
            return dataset, marker[1:] + suffix
    raise ValueError(f"Cannot identify dataset and model from {path}")


def parse_raw_predictions(frame, label_column):
    """Reproduce the parser used by eval_gemma_zero_shot.py."""
    possible_labels = [str(value) for value in frame[label_column].dropna().unique()]
    possible_labels_lower = [value.lower() for value in possible_labels]
    parsed = []
    for prediction in frame["gemma_prediction"]:
        prediction_lower = str(prediction).lower()
        matched = possible_labels[0]
        for label, label_lower in zip(possible_labels, possible_labels_lower):
            if label_lower in prediction_lower:
                matched = label
                break
        parsed.append(matched)
    return np.asarray(parsed, dtype=str)


def load_predictions(path, dataset):
    label_column = LABEL_COLUMNS[dataset]
    columns = pd.read_csv(path, nrows=0).columns
    usecols = [label_column, "gemma_prediction"]
    if "gemma_parsed_prediction" in columns:
        usecols.append("gemma_parsed_prediction")
    frame = pd.read_csv(path, usecols=usecols)
    y_true = frame[label_column].astype(str).to_numpy()
    if "gemma_parsed_prediction" in frame:
        y_pred = frame["gemma_parsed_prediction"].astype(str).to_numpy()
    else:
        y_pred = parse_raw_predictions(frame, label_column)
    return y_true, y_pred


def metrics_from_codes(y_true, y_pred, n_labels):
    confusion = np.bincount(
        y_true * n_labels + y_pred, minlength=n_labels * n_labels
    ).reshape(n_labels, n_labels)
    true_positive = np.diag(confusion)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    present = denominator > 0
    per_class_f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros(n_labels, dtype=float),
        where=present,
    )
    return np.trace(confusion) / confusion.sum(), per_class_f1[present].mean()


def bootstrap_metrics(y_true, y_pred, samples, seed):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true_codes = np.fromiter((label_to_index[value] for value in y_true), dtype=int)
    y_pred_codes = np.fromiter((label_to_index[value] for value in y_pred), dtype=int)
    n_labels = len(labels)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accuracy_draws = np.empty(samples)
    macro_f1_draws = np.empty(samples)
    for index in range(samples):
        sampled = rng.integers(0, n, n)
        accuracy_draws[index], macro_f1_draws[index] = metrics_from_codes(
            y_true_codes[sampled], y_pred_codes[sampled], n_labels
        )
    accuracy, macro_f1 = metrics_from_codes(y_true_codes, y_pred_codes, n_labels)
    return {
        "accuracy": (
            accuracy,
            np.quantile(accuracy_draws, 0.025),
            np.quantile(accuracy_draws, 0.975),
        ),
        "macro_f1": (
            macro_f1,
            np.quantile(macro_f1_draws, 0.025),
            np.quantile(macro_f1_draws, 0.975),
        ),
    }


def write_latex(summary, path):
    order = {"Recipes5k": 0, "fakeddit": 1, "mbrset": 2}
    wide = summary.pivot(index=["dataset", "model", "n_test"], columns="metric")
    rows = []
    for (dataset, model, n_test), values in wide.iterrows():
        accuracy = values["estimate"]["accuracy"]
        accuracy_low = values["ci_low"]["accuracy"]
        accuracy_high = values["ci_high"]["accuracy"]
        macro_f1 = values["estimate"]["macro_f1"]
        macro_f1_low = values["ci_low"]["macro_f1"]
        macro_f1_high = values["ci_high"]["macro_f1"]
        rows.append(
            (
                order[dataset],
                model,
                f"{dataset} & {MODEL_LABELS.get(model, model)} & {n_test:,} & "
                f"{accuracy:.3f} [{accuracy_low:.3f}, {accuracy_high:.3f}] & "
                f"{macro_f1:.3f} [{macro_f1_low:.3f}, {macro_f1_high:.3f}] \\\\",
            )
        )
    rows.sort(key=lambda row: (row[0], row[1]))
    with open(path, "w", encoding="utf-8") as output:
        output.write("\\begin{tabular}{llrcc}\n")
        output.write("\\toprule\n")
        output.write("Dataset & Model & $n$ & Accuracy [95\\% CI] & Macro-F1 [95\\% CI] \\\\\n")
        output.write("\\midrule\n")
        output.write("\n".join(row[2] for row in rows))
        output.write("\n\\bottomrule\n\\end{tabular}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", default="outputs/gemma_zero_shot")
    parser.add_argument("--output_dir", default="outputs/gemma_zero_shot")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected_cases", type=int, default=8)
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.predictions_dir, "*_predictions.csv")))
    if not paths:
        raise FileNotFoundError(f"No model-specific prediction files in {args.predictions_dir}")
    if len(paths) != args.expected_cases and not args.allow_partial:
        raise RuntimeError(
            f"Expected {args.expected_cases} prediction files, found {len(paths)}. "
            "Use --allow_partial only for an explicit partial test."
        )

    rows = []
    for path in paths:
        dataset, model = identify_case(path)
        if dataset not in LABEL_COLUMNS:
            continue
        y_true, y_pred = load_predictions(path, dataset)
        metrics = bootstrap_metrics(y_true, y_pred, args.samples, args.seed)
        for metric, (estimate, ci_low, ci_high) in metrics.items():
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "bootstrap_samples": args.samples,
                    "n_test": len(y_true),
                    "prediction_file": os.path.basename(path),
                }
            )

    os.makedirs(args.output_dir, exist_ok=True)
    summary = pd.DataFrame(rows).sort_values(["dataset", "model", "metric"])
    csv_path = os.path.join(args.output_dir, "mllm_bootstrap_summary.csv")
    tex_path = os.path.join(args.output_dir, "mllm_bootstrap_appendix.tex")
    summary.to_csv(csv_path, index=False)
    write_latex(summary, tex_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
