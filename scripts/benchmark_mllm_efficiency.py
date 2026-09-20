"""End-to-end inference efficiency benchmark for configured direct MLLMs.

This is deliberately independent from the frozen-embedding fusion workflow.
It uses the same held-out split, prompt template, model wrapper, quantization,
and deterministic generation settings as ``eval_gemma_zero_shot.py``.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_utils import split_data
from src.models import GemmaVLM


def build_prompt(dataset, labels, text):
    prompt = "Task: Answer the following question based on the image.\n"
    if dataset.lower() == "fakeddit":
        prompt += "This is a single-label task. Choose exactly one of the following classes: 0 (Real) or 1 (Fake).\n"
    elif dataset.lower() in ("mbrset", "brset"):
        prompt += "This is a single-label task. Choose exactly one of the following classes: 0 (No Diabetic Retinopathy) or 1 (Diabetic Retinopathy).\n"
    else:
        prompt += f"This is a single-label task. Choose exactly one of the following classes: {labels}.\n"
    return f"{prompt}Question/Text: {text}\nAnswer directly with the class name(s)."


def load_case(config, index):
    entries = [item for item in config["gemma_zero_shot"]["datasets"] if item.get("enabled", False)]
    if index < 0 or index >= len(entries):
        raise IndexError(f"case_index must be in [0, {len(entries) - 1}]")
    return entries[index], entries


def resolve_cached_snapshot(model_id):
    """Return a complete local snapshot from either HF cache layout.

    The original zero-shot runs used ``$HF_HOME/models--...``.  Newer
    Hugging Face versions default to ``$HF_HOME/hub/models--...``.  Check
    both layouts explicitly and pass the snapshot path to Transformers, so a
    benchmark cannot fetch model files or accidentally use an incomplete
    cache entry.
    """
    cache_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    repository = "models--" + model_id.replace("/", "--")
    repositories = (cache_home / repository, cache_home / "hub" / repository)
    failures = []
    for repo in repositories:
        ref = repo / "refs" / "main"
        if not ref.is_file():
            failures.append(f"{repo}: no refs/main")
            continue
        snapshot = repo / "snapshots" / ref.read_text().strip()
        index = snapshot / "model.safetensors.index.json"
        if not index.is_file():
            failures.append(f"{snapshot}: no safetensors index")
            continue
        weight_files = set(json.loads(index.read_text())["weight_map"].values())
        missing = sorted(name for name in weight_files if not (snapshot / name).is_file())
        if missing:
            failures.append(f"{snapshot}: missing {', '.join(missing[:3])}")
            continue
        return snapshot
    details = "\n  ".join(failures)
    raise FileNotFoundError(
        f"No complete local snapshot found for {model_id}. Checked:\n  {details}"
    )


def main():
    parser = argparse.ArgumentParser(description="Profile configured Gemma/MedGemma zero-shot inference.")
    parser.add_argument("--config", default="configs/paper_experiments.yaml")
    parser.add_argument("--case_index", type=int, required=True)
    parser.add_argument("--samples", type=int, default=20, help="Fixed held-out samples for the inference microbenchmark.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=16, help="Classification answers are short; this bounds generation consistently.")
    parser.add_argument("--output_dir", default="outputs/mllm_efficiency")
    parser.add_argument("--allow_download", action="store_true", help="Resume/download a missing checkpoint from Hugging Face.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    case, all_cases = load_case(config, args.case_index)
    paths = config.get("paths", {})
    embedding_path = os.path.join(paths.get("embeddings_dir", "Embeddings_vlm"), case["embeddings_subdir"], f"embeddings_{case.get('backbone', 'clip')}.csv")
    df = pd.read_csv(embedding_path)
    label_col, text_col, image_col = case["label_col"], case["text_col"], case["image_col"]
    df = df.dropna(subset=[label_col, text_col, image_col])
    splits = split_data(df, val_size=0.1, random_state=42)
    test_df = splits[2] if len(splits) == 3 else splits[1]
    selected = test_df.sample(n=min(args.samples, len(test_df)), random_state=42)
    labels = test_df[label_col].dropna().unique().tolist()

    quantization = config["gemma_zero_shot"].get("quantization", "8b")
    try:
        snapshot_path = resolve_cached_snapshot(case["model_name"])
        model_id = str(snapshot_path)
    except FileNotFoundError:
        if not args.allow_download:
            raise
        # The original 27B checkpoint is not complete in the current cache.
        # An explicit flag is required before Transformers can resume it.
        snapshot_path = None
        model_id = case["model_name"]
        print(f"No complete local snapshot for {model_id}; allowing an explicit Hugging Face download/resume.")
    load_start = time.perf_counter()
    # Cache-only loading prevents a profiling job from silently downloading
    # weights instead of measuring inference.
    model = GemmaVLM(model_id, quantization=quantization, offline_mode=not args.allow_download)
    load_seconds = time.perf_counter() - load_start
    parameter_count = sum(param.numel() for param in model.model.parameters())
    parameter_tensor_mb = sum(param.numel() * param.element_size() for param in model.model.parameters()) / 1024 ** 2

    def prepare(row):
        path = row[image_col]
        image = Image.open(path).convert("RGB") if isinstance(path, str) and os.path.exists(path) else None
        return build_prompt(case["name"], labels, row[text_col]), image

    prepared = [prepare(row) for _, row in selected.iterrows()]
    with torch.inference_mode():
        for prompt, image in prepared[:min(args.warmup, len(prepared))]:
            model.generate_batch([prompt], [image], max_new_tokens=args.max_new_tokens, temperature=0.01, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies, output_tokens = [], []
    with torch.inference_mode():
        for prompt, image in prepared:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start = time.perf_counter()
            output = model.generate_batch([prompt], [image], max_new_tokens=args.max_new_tokens, temperature=0.01, do_sample=False)[0]
            if torch.cuda.is_available(): torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
            tokenizer = getattr(model.processor, "tokenizer", model.processor)
            try:
                output_tokens.append(len(tokenizer.encode(output, add_special_tokens=False)))
            except Exception:
                output_tokens.append(np.nan)
    os.makedirs(args.output_dir, exist_ok=True)
    row = {
        "dataset": case["name"], "model": case["model_name"].split("/")[-1], "case_index": args.case_index,
        "model_source": str(snapshot_path) if snapshot_path else case["model_name"],
        "quantization": quantization, "n_profiled": len(prepared), "warmup": args.warmup,
        "max_new_tokens": args.max_new_tokens, "model_load_seconds": load_seconds,
        "parameter_count": parameter_count, "parameter_tensor_mb": parameter_tensor_mb,
        "latency_p50_ms": np.percentile(latencies, 50) * 1000, "latency_p95_ms": np.percentile(latencies, 95) * 1000,
        "throughput_samples_per_s": len(latencies) / sum(latencies),
        "throughput_output_tokens_per_s": np.nansum(output_tokens) / sum(latencies),
        "mean_output_tokens": np.nanmean(output_tokens),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else np.nan,
    }
    stem = f"{case['name']}_{row['model']}_efficiency"
    pd.DataFrame([row]).to_csv(os.path.join(args.output_dir, f"{stem}.csv"), index=False)
    print(pd.Series(row).to_string())


if __name__ == "__main__":
    main()
