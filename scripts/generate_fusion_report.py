"""Build paper-ready tables and a Pareto figure from completed experiment CSVs."""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


METHOD_ORDER = ["image_only", "text_only", "linear", "mlp_early", "mlp_late", "gated_attention", "mcr_late", "i2moe"]
METHOD_LABELS = {"image_only": "Image-only probe", "text_only": "Text-only probe", "linear": "Linear concat", "mlp_early": "Early MLP", "mlp_late": "Late MLP", "gated_attention": "Gated attention", "mcr_late": "MCR late", "i2moe": "I²MoE"}
COLORS = {"image_only": "#8c8c8c", "text_only": "#bdbdbd", "linear": "#1f77b4", "mlp_early": "#ff7f0e", "mlp_late": "#2ca02c", "gated_attention": "#d62728", "mcr_late": "#9467bd", "i2moe": "#17becf"}


def ci_string(row):
    return f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/rebuttal_fusion")
    parser.add_argument("--mllm_efficiency_csv", default="outputs/mllm_efficiency/mllm_performance_efficiency.csv")
    args = parser.parse_args()
    out = os.path.join(args.results_dir, "final")
    os.makedirs(out, exist_ok=True)
    metrics = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(args.results_dir, "*_results.csv"))], ignore_index=True)
    metrics["method"] = pd.Categorical(metrics.method, METHOD_ORDER, ordered=True)
    boot = pd.read_csv(os.path.join(args.results_dir, "bootstrap_summary.csv"))
    pivot = boot.pivot(index=["dataset", "backbone", "method"], columns="metric", values=["estimate", "ci_low", "ci_high"])
    pivot.columns = ["_".join(c) for c in pivot.columns]
    table = metrics.merge(pivot.reset_index(), on=["dataset", "backbone", "method"], validate="one_to_one")
    table["method_label"] = table.method.map(METHOD_LABELS)
    for metric, title in [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]:
        table[title] = table[[f"estimate_{metric}", f"ci_low_{metric}", f"ci_high_{metric}"]].rename(columns={f"estimate_{metric}": "estimate", f"ci_low_{metric}": "ci_low", f"ci_high_{metric}": "ci_high"}).apply(ci_string, axis=1)
    performance = table[["dataset", "backbone", "method_label", "Accuracy", "Macro-F1", "epochs_run", "n_test"]].sort_values(["dataset", "backbone", "method_label"])
    performance.to_csv(os.path.join(out, "performance_with_bootstrap_ci.csv"), index=False)
    performance.to_latex(os.path.join(out, "performance_with_bootstrap_ci.tex"), index=False, escape=False)

    # Direct zero-shot MLLM results are reported separately: their end-to-end
    # inference cost is not comparable with a head operating on cached embeddings.
    zero_shot_dir = os.path.join(os.path.dirname(args.results_dir), "gemma_zero_shot")
    zero_shot_paths = glob.glob(os.path.join(zero_shot_dir, "*_Zero-Shot_results.csv"))
    if zero_shot_paths:
        zero_shot = pd.concat([pd.read_csv(path) for path in zero_shot_paths], ignore_index=True)
        zero_shot = zero_shot[["dataset", "backbone", "fusion_method", "accuracy", "f1_score"]].rename(
            columns={"backbone": "model", "fusion_method": "evaluation", "f1_score": "macro_f1"}
        )
        zero_shot["model_family"] = "Direct zero-shot MLLM"
        zero_shot = zero_shot[["dataset", "model_family", "model", "evaluation", "accuracy", "macro_f1"]]
        zero_shot.to_csv(os.path.join(out, "zero_shot_mllm_performance.csv"), index=False)

        fusion_for_all = metrics[["dataset", "backbone", "method", "accuracy", "f1_score"]].copy()
        fusion_for_all["model_family"] = "Frozen-embedding fusion"
        fusion_for_all = fusion_for_all.rename(columns={"backbone": "model", "method": "evaluation", "f1_score": "macro_f1"})
        all_models = pd.concat([fusion_for_all, zero_shot], ignore_index=True, sort=False)
        all_models.to_csv(os.path.join(out, "all_model_performance.csv"), index=False)

    efficiency = pd.read_csv(os.path.join(args.results_dir, "efficiency.csv"))
    combined = table.merge(efficiency, on=["dataset", "backbone", "method"], validate="one_to_one")
    eff_table = combined[["dataset", "backbone", "method_label", "estimate_macro_f1", "trainable_parameters", "model_size_mb", "flops_per_sample", "latency_p50_ms", "latency_p95_ms", "peak_gpu_memory_mb"]].rename(columns={"estimate_macro_f1": "macro_f1"}).sort_values(["dataset", "backbone", "method_label"])
    eff_table.to_csv(os.path.join(out, "efficiency_performance.csv"), index=False)
    eff_table.to_latex(os.path.join(out, "efficiency_performance.tex"), index=False, float_format="%.4g", escape=False)

    # A single, grouped performance--efficiency table keeps the two inference
    # routes together while preserving their different cost definitions.
    fusion_rows = combined[["dataset", "backbone", "method_label", "estimate_accuracy", "estimate_macro_f1", "trainable_parameters", "flops_per_sample", "latency_p50_ms", "peak_gpu_memory_mb"]].copy()
    fusion_rows.insert(0, "model_family", "Embedding fusion")
    fusion_rows = fusion_rows.rename(columns={
        "backbone": "model", "method_label": "method",
        "estimate_accuracy": "accuracy", "estimate_macro_f1": "macro_f1",
    })
    if os.path.isfile(args.mllm_efficiency_csv):
        mllm = pd.read_csv(args.mllm_efficiency_csv)
        mllm_rows = mllm[["dataset", "model", "accuracy", "macro_f1", "parameter_count", "latency_p50_ms", "peak_gpu_memory_mb"]].copy()
        mllm_rows.insert(0, "model_family", "Direct MLLM")
        mllm_rows.insert(3, "method", "Zero-shot")
        mllm_rows = mllm_rows.rename(columns={"parameter_count": "trainable_parameters"})
        mllm_rows["flops_per_sample"] = pd.NA
        shared_columns = ["model_family", "dataset", "model", "method", "accuracy", "macro_f1", "trainable_parameters", "flops_per_sample", "latency_p50_ms", "peak_gpu_memory_mb"]
        full_table = pd.concat([mllm_rows[shared_columns], fusion_rows[shared_columns]], ignore_index=True)
        full_table["model_family"] = pd.Categorical(full_table["model_family"], ["Direct MLLM", "Embedding fusion"], ordered=True)
        full_table = full_table.sort_values(["model_family", "dataset", "model", "method"])
        full_table.to_csv(os.path.join(out, "combined_performance_efficiency.csv"), index=False)

    paired = pd.read_csv(os.path.join(args.results_dir, "bootstrap_pairwise.csv"))
    paired.to_csv(os.path.join(out, "paired_bootstrap_all.csv"), index=False)

    settings = list(combined[["dataset", "backbone"]].drop_duplicates().itertuples(index=False, name=None))

    def cost_figure(x_column, y_column, x_label, y_label, stem, log_x=False):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ax, (dataset, backbone) in zip(axes.flat, settings):
            data = combined[(combined.dataset == dataset) & (combined.backbone == backbone)].sort_values("method")
            for row in data.itertuples():
                ax.scatter(getattr(row, x_column), getattr(row, y_column), s=55, color=COLORS[row.method], zorder=3)
            ax.set_title(f"{dataset} / {backbone}")
            ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.grid(alpha=.25)
            if log_x: ax.set_xscale("log")
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[method], markersize=7, label=METHOD_LABELS[method]) for method in METHOD_ORDER]
        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.07), frameon=False)
        fig.savefig(os.path.join(out, f"{stem}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(out, f"{stem}.pdf"), bbox_inches="tight")

    cost_figure("latency_p50_ms", "estimate_macro_f1", "p50 latency (ms/sample)", "Macro-F1", "pareto_macro_f1_latency")
    cost_figure("flops_per_sample", "estimate_accuracy", "FLOPs/sample", "Accuracy", "pareto_accuracy_flops", log_x=True)
    cost_figure("trainable_parameters", "estimate_macro_f1", "Trainable parameters", "Macro-F1", "pareto_macro_f1_parameters", log_x=True)
    cost_figure("peak_gpu_memory_mb", "estimate_macro_f1", "Peak GPU memory (MB)", "Macro-F1", "pareto_macro_f1_memory")

    # Combined space-efficient view: MLLMs are end-to-end measurements; fusion
    # points are head-only measurements from precomputed embeddings.
    if os.path.isfile(args.mllm_efficiency_csv):
        mllm = pd.read_csv(args.mllm_efficiency_csv)
        datasets = ["Recipes5k", "fakeddit", "mbrset"]
        backbone_markers = {"clip": "o", "siglip": "s", "biomedclip": "D", "medsiglip": "^"}
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
        for ax, dataset in zip(axes, datasets):
            fusion_data = combined[combined.dataset == dataset]
            for row in fusion_data.itertuples():
                ax.scatter(row.latency_p50_ms, row.estimate_macro_f1, s=48,
                           marker=backbone_markers[row.backbone], color=COLORS[row.method],
                           edgecolor="white", linewidth=.35, zorder=3)
            direct_data = mllm[mllm.dataset == dataset]
            ax.scatter(direct_data.latency_p50_ms, direct_data.macro_f1, s=105,
                       marker="*", color="#111111", edgecolor="white", linewidth=.45, zorder=4)
            ax.set_title(dataset)
            ax.set_xscale("log")
            ax.set_xlabel("p50 latency (ms/sample; log scale)")
            ax.set_ylabel("Macro-F1")
            ax.grid(alpha=.25, which="both")
        method_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[m], markersize=6, label=METHOD_LABELS[m]) for m in METHOD_ORDER]
        backbone_handles = [Line2D([0], [0], marker=marker, color="#555555", linestyle="None", markersize=6, label=backbone) for backbone, marker in backbone_markers.items()]
        mllm_handle = Line2D([0], [0], marker="*", color="#111111", linestyle="None", markersize=9, label="Direct MLLM")
        fig.legend(handles=method_handles + backbone_handles + [mllm_handle], loc="lower center", ncol=5, bbox_to_anchor=(.5, -.16), frameon=False)
        fig.savefig(os.path.join(out, "combined_macro_f1_latency.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(out, "combined_macro_f1_latency.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
