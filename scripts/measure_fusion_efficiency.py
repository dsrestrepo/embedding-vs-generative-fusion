"""Inference-only efficiency benchmark for saved frozen-embedding fusion heads."""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.fusion_models import build_model

DATASETS = {"Recipes5k": ("Recipes5k", "class"), "fakeddit": ("fakeddit", "2_way_label"), "mbrset": ("mbrset", "DR_2")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/rebuttal_fusion")
    parser.add_argument("--embeddings_dir", default="/gpfs/workdir/restrepoda/Embeddings_vlm")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for metric_path in glob.glob(os.path.join(args.results_dir, "*_results.csv")):
        metric = pd.read_csv(metric_path).iloc[0]; dataset, backbone, method = metric.dataset, metric.backbone, metric.method
        subdir, label = DATASETS[dataset]
        embed = pd.read_csv(os.path.join(args.embeddings_dir, subdir, f"embeddings_{backbone}.csv"), nrows=1)
        text_cols = [c for c in embed if c.startswith("text_emb_")]; image_cols = [c for c in embed if c.startswith("img_emb_")]
        model = build_model(method, len(text_cols), len(image_cols), int(metric.n_classes))
        state = torch.load(metric_path.replace("_results.csv", "_model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(state); model.to(device).eval()
        text, image = torch.randn(1, len(text_cols), device=device), torch.randn(1, len(image_cols), device=device)
        with torch.inference_mode():
            for _ in range(args.warmup): model(text, image)
            if device.type == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(args.iterations):
                if device.type == "cuda":
                    start, end = torch.cuda.Event(True), torch.cuda.Event(True); start.record(); model(text, image); end.record(); torch.cuda.synchronize(); times.append(start.elapsed_time(end))
                else:
                    import time; start = time.perf_counter(); model(text, image); times.append((time.perf_counter()-start)*1000)
        # PyTorch profiler reports FLOPs for supported operators (primarily Linear).
        with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []), with_flops=True) as prof:
            model(text, image)
        flops = sum(event.flops for event in prof.key_averages())
        rows.append({"dataset": dataset, "backbone": backbone, "method": method,
                     "trainable_parameters": sum(p.numel() for p in model.parameters()),
                     "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2,
                     "flops_per_sample": flops, "latency_p50_ms": np.percentile(times, 50), "latency_p95_ms": np.percentile(times, 95),
                     "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else np.nan),
                     "device": str(device), "warmup": args.warmup, "iterations": args.iterations})
    pd.DataFrame(rows).to_csv(os.path.join(args.results_dir, "efficiency.csv"), index=False)


if __name__ == "__main__": main()
