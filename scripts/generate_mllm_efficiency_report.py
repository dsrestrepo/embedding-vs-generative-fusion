"""Aggregate direct-MLLM task performance with isolated efficiency profiles."""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--efficiency_dir", default="outputs/mllm_efficiency")
    parser.add_argument("--zero_shot_dir", default="outputs/gemma_zero_shot")
    args = parser.parse_args()
    profiles = [
        path for path in glob.glob(os.path.join(args.efficiency_dir, "*_efficiency.csv"))
        if os.path.basename(path) != "mllm_performance_efficiency.csv"
    ]
    if not profiles:
        raise FileNotFoundError("No MLLM efficiency profiles found.")
    efficiency = pd.concat([pd.read_csv(path) for path in profiles], ignore_index=True)
    metric_paths = glob.glob(os.path.join(args.zero_shot_dir, "*_Zero-Shot_results.csv"))
    performance = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    # ``model`` is the broad family label in the saved zero-shot files, while
    # ``backbone`` is the concrete checkpoint identifier used by efficiency.
    performance = performance.drop(columns=["model"], errors="ignore").rename(
        columns={"backbone": "model", "f1_score": "macro_f1"}
    )
    report = efficiency.merge(performance[["dataset", "model", "accuracy", "macro_f1"]], on=["dataset", "model"], validate="one_to_one")
    report = report.sort_values(["dataset", "model"])
    report.to_csv(os.path.join(args.efficiency_dir, "mllm_performance_efficiency.csv"), index=False)
    report.to_latex(os.path.join(args.efficiency_dir, "mllm_performance_efficiency.tex"), index=False, float_format="%.4g")

    datasets = list(report.dataset.unique())
    unique_models = list(dict.fromkeys(report.model.str.replace("-it", "", regex=False)))
    colors = {model: f"C{i}" for i, model in enumerate(unique_models)}
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), constrained_layout=True)
    if len(datasets) == 1: axes = [axes]
    for ax, dataset in zip(axes, datasets):
        data = report[report.dataset == dataset]
        for row in data.itertuples():
            model = row.model.replace("-it", "")
            ax.scatter(row.latency_p50_ms, row.macro_f1, s=60, color=colors[model])
        ax.set_title(dataset); ax.set_xlabel("End-to-end p50 latency (ms/sample)"); ax.set_ylabel("Macro-F1"); ax.grid(alpha=.25)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[model], markersize=7, label=model) for model in unique_models]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.08), frameon=False)
    fig.savefig(os.path.join(args.efficiency_dir, "mllm_macro_f1_latency.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(args.efficiency_dir, "mllm_macro_f1_latency.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
