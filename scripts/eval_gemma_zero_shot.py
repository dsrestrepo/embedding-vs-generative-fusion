import os
import sys
import argparse
import pandas as pd
from typing import Any, Dict, List

import yaml

# Ensure the root directory is on the path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import GemmaVLM
from src.data_utils import split_data
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def enabled_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in items if item.get("enabled", False)]


def run_evaluation(args):
    missing = [
        name
        for name in ("embeddings_csv", "label_col")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(f"Missing required arguments for single run: {', '.join(missing)}")

    # Load data
    df = pd.read_csv(args.embeddings_csv)
    
    # Special DAQUAR handling as per reference file
    if args.dataset.lower() == 'daquar':
        # Need to split data first to find test set valid classes
        pass # We'll do this after splitting
    
    # Special MIMIC handling as per reference file
    if args.dataset.lower() == 'mimic' and args.multilabel:
        disease_cols = args.label_col
        df = df[(df[disease_cols] != -1).all(axis=1)]
        df[disease_cols] = df[disease_cols].fillna(0)

    # Attempt to fall back to common column names if user-specified ones don't exist
    available_cols = df.columns.tolist()
    if args.text_col not in available_cols:
        possible_texts = ['text', 'question', 'questions', 'ingredients', 'text_emb_original']
        for c in possible_texts:
            if c in available_cols:
                args.text_col = c
                break

    if args.image_col not in available_cols:
        possible_images = ['image_path', 'image', 'image_id', 'id', 'path_preproc', 'file', 'img_path']
        for c in possible_images:
            if c in available_cols:
                args.image_col = c
                break

    df = df.dropna(subset=args.label_col + [args.text_col, args.image_col])

    # Split data to get the exact same test set as the framework fusion
    splits = split_data(df, val_size=0.1, random_state=42)
    if len(splits) == 3:
        train_df, val_df, test_df = splits
    else:
        train_df, test_df = splits
        
    print(f"Initial test set with {len(test_df)} samples.")
    
    # DAQUAR Test Set Filtering (Only keeping classes with >= 20 samples in test set)
    if args.dataset.lower() == 'daquar':
        print(f"Applying {args.dataset} filter: keeping classes with >= 20 samples in test set.")
        label_column = args.label_col[0]
        
        temp_test_series = test_df[label_column]
        first_val = temp_test_series.iloc[0] if len(temp_test_series) > 0 else ""
        if isinstance(first_val, str) and ',' in first_val:
            temp_test_series = temp_test_series.str.split(',').explode().str.strip()
        
        label_counts = temp_test_series.value_counts()
        valid_classes = label_counts[label_counts >= 20].index.tolist()
        
        print(f"Found {len(valid_classes)} valid classes with >= 20 samples in test set.")
        
        def filter_rows(row, valid):
            val = row[label_column]
            if isinstance(val, str):
                parts = [p.strip() for p in val.split(',')]
                return any(p in valid for p in parts)
            return val in valid

        test_df = test_df[test_df.apply(lambda row: filter_rows(row, valid_classes), axis=1)]
        print(f"Filtered test set with {len(test_df)} samples.")

    print(f"Loaded final test set with {len(test_df)} samples.")

    # Get possible labels and extract ground truths
    if len(args.label_col) > 1:
        # Multi-column labels (e.g., MIMIC)
        possible_labels = args.label_col
        ground_truths = test_df[args.label_col].values.tolist()
    else:
        # Single column
        label_prop = args.label_col[0]
        if args.multilabel:
            # Assumes comma separated strings or lists for things like Daquar
            all_labels = test_df[label_prop].dropna().astype(str).str.split(',').explode().str.strip()
            possible_labels = all_labels.unique().tolist()
        else:
            possible_labels = test_df[label_prop].dropna().unique().tolist()
        ground_truths = test_df[label_prop].tolist()

    print(f"Possible labels: {possible_labels}")

    # Initialize model in online mode so it can download weights
    model = GemmaVLM(args.model_name, quantization=args.quantization, offline_mode=False)

    predictions = []
    
    # Process in batches
    batch_prompts = []
    batch_images = []
    count = 0
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        count += 1
        image_path = row[args.image_col]
        text_prompt = row[args.text_col]
        
        # Build prompt
        prompt = f"Task: Answer the following question based on the image.\n"
        if args.multilabel:
            prompt += f"This is a multilabel task. Possible classes are: {possible_labels}. Select all that apply.\n"
        else:
            if args.dataset.lower() == 'fakeddit':
                prompt += "This is a single-label task. Choose exactly one of the following classes: 0 (Real) or 1 (Fake).\n"
            elif args.dataset.lower() in ['mbrset', 'brset']:
                prompt += "This is a single-label task. Choose exactly one of the following classes: 0 (No Diabetic Retinopathy) or 1 (Diabetic Retinopathy).\n"
            else:
                prompt += f"This is a single-label task. Choose exactly one of the following classes: {possible_labels}.\n"
        prompt += f"Question/Text: {text_prompt}\nAnswer directly with the class name(s)."

        batch_prompts.append(prompt)
        batch_images.append(image_path)
        
        if len(batch_prompts) == args.batch_size or count == len(test_df):
            try:
                pil_images = []
                for p in batch_images:
                    if isinstance(p, str) and os.path.exists(p):
                        pil_images.append(Image.open(p).convert("RGB"))
                    else:
                        pil_images.append(None)
                        
                preds = model.generate_batch(prompts=batch_prompts, images=pil_images)
                predictions.extend(preds)
            except Exception as e:
                print(f"Error on batch: {e}")
                predictions.extend([""] * len(batch_prompts))
            
            # Clear batch
            batch_prompts = []
            batch_images = []

    # Evaluate using the exact same metrics (Acc & F1)
    parsed_preds = []
    if len(args.label_col) > 1:
        # Multi-column Multilabel (MIMIC)
        possible_labels_lower = [l.lower() for l in possible_labels]
        for p in predictions:
            p_lower = p.lower()
            parsed = [1 if lbl in p_lower else 0 for lbl in possible_labels_lower]
            parsed_preds.append(parsed)
        y_true = np.array(ground_truths)
        y_pred = np.array(parsed_preds)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
    elif args.multilabel:
        # Single-column Multilabel (Daquar)
        mlb = MultiLabelBinarizer()
        
        # Parse ground truths (comma separated)
        parsed_gt = [str(gt).split(',') for gt in ground_truths]
        parsed_gt = [[lbl.strip() for lbl in lst] for lst in parsed_gt]
        y_true = mlb.fit_transform(parsed_gt)
        
        # Parse predictions against fit classes
        possible_labels_lower = [str(c).lower() for c in mlb.classes_]
        for p in predictions:
            p_lower = str(p).lower()
            parsed = [1 if lbl in p_lower else 0 for lbl in possible_labels_lower]
            parsed_preds.append(parsed)
        y_pred = np.array(parsed_preds)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
    else:
        # Single-label classification
        possible_labels_str = [str(l) for l in possible_labels]
        possible_labels_lower = [l.lower() for l in possible_labels_str]
        for p in predictions:
            p_lower = str(p).lower()
            # Find first matching label
            matched = possible_labels_str[0] # Default fallback
            for i, lbl in enumerate(possible_labels_lower):
                if lbl in p_lower:
                    matched = possible_labels_str[i]
                    break
            parsed_preds.append(matched)
            
        # Ensure y_true and y_pred are exactly the same type (strings) to prevent sklearn ValueError
        y_true = np.array([str(x) for x in ground_truths])
        y_pred = np.array(parsed_preds)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

    print("\n========= METRICS =========")
    print(f"Dataset: {args.dataset} | Method: Zero-Shot LLM")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")

    # Output raw predictions to file
    test_df['gemma_prediction'] = predictions
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file_preds = f"{output_dir}/{args.dataset}_zero_shot_preds.csv"
    test_df.to_csv(output_file_preds, index=False)
    
    # Save framework-compatible metrics file
    metrics = {
        'model': 'Zero-Shot LLM',
        'dataset': args.dataset,
        'backbone': args.model_name.split('/')[-1],  # Extract e.g. medgemma-27b-it
        'fusion_method': 'Zero-Shot',
        'accuracy': acc,
        'f1_score': f1,
        'auc': 0.0  # Zero-shot LLM typically doesn't map cleanly to AUC probabilities natively
    }
    output_file_metrics = f"{output_dir}/{args.dataset}_{metrics['backbone']}_Zero-Shot_results.csv"
    pd.DataFrame([metrics]).to_csv(output_file_metrics, index=False)
    
    print(f"Predictions saved to {output_file_preds}")
    print(f"Metrics saved to {output_file_metrics}")


def run_configured_evaluations(config_path: str):
    config = load_config(config_path)
    paths = config.get("paths", {})
    embeddings_dir = paths.get("embeddings_dir", "Embeddings_vlm")

    zero_shot_cfg = config.get("gemma_zero_shot", {})
    output_dir = zero_shot_cfg.get("output_dir", "outputs/gemma_zero_shot")
    datasets = enabled_items(zero_shot_cfg.get("datasets", []))

    if not datasets:
        print("No enabled datasets found in gemma_zero_shot config.")
        return

    for dataset in datasets:
        backbone = dataset.get("backbone", "clip")
        embeddings_csv = os.path.join(
            embeddings_dir,
            dataset["embeddings_subdir"],
            f"embeddings_{backbone}.csv",
        )
        args = argparse.Namespace(
            model_name=dataset["model_name"],
            embeddings_csv=embeddings_csv,
            label_col=(
                [dataset["label_col"]]
                if isinstance(dataset["label_col"], str)
                else dataset["label_col"]
            ),
            text_col=dataset["text_col"],
            image_col=dataset["image_col"],
            dataset=dataset["name"],
            multilabel=dataset.get("multilabel", False),
            batch_size=zero_shot_cfg.get("batch_size", 4),
            quantization=zero_shot_cfg.get("quantization", "8b"),
            output_dir=output_dir,
        )
        print(f"--- Running Zero-Shot on {args.dataset} ({args.model_name}) ---")
        run_evaluation(args)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation with Gemma models.")
    parser.add_argument("--config", type=str, help="Run enabled gemma_zero_shot entries from a YAML config")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it", help="Model name or path")
    parser.add_argument("--embeddings_csv", type=str, help="Path to pre-extracted embeddings (contains paths and labels)")
    parser.add_argument("--label_col", type=str, nargs='+', help="Column name(s) with ground truth label")
    parser.add_argument("--text_col", type=str, default="text", help="Column name with text/question")
    parser.add_argument("--image_col", type=str, default="image_path", help="Column name with image path")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset name")
    parser.add_argument("--multilabel", action="store_true", help="Set flag if classification is multilabel")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for model evaluation")
    parser.add_argument("--quantization", type=str, default="8b", help="Quantization format to use (16b, 8b, 4b)")
    parser.add_argument("--output_dir", type=str, default="outputs/gemma_zero_shot", help="Output folder")
    args = parser.parse_args()

    if args.config:
        run_configured_evaluations(args.config)
    else:
        run_evaluation(args)

if __name__ == '__main__':
    main()
