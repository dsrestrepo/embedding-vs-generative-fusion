import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import os

def main():
    preds_file = "outputs/gemma_zero_shot/fakeddit_zero_shot_preds.csv"
    metrics_file = "outputs/gemma_zero_shot/fakeddit_gemma-3-27b-it_Zero-Shot_results.csv"

    print(f"Reading {preds_file}...")
    df = pd.read_csv(preds_file)
    
    y_true = df["2_way_label"].astype(int).values

    def parse_prediction(p):
        p = str(p).lower()
        # Legacy predictions used the inverted prompt "0 (Real), 1 (Fake)".
        # Convert those semantic answers to the dataset mapping: 0 = Fake, 1 = Real.
        if '0' in p or 'real' in p:
            return 1
        elif '1' in p or 'fake' in p:
            return 0
        else:
            return 0 # Fallback 

    y_pred = [parse_prediction(p) for p in df["gemma_prediction"]]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Corrected Accuracy: {acc:.4f}")
    print(f"Corrected F1 (Macro): {f1:.4f}")

    # Write to metrics file
    metrics = {
        'model': 'Zero-Shot LLM',
        'dataset': 'fakeddit',
        'backbone': 'gemma-3-27b-it',
        'fusion_method': 'Zero-Shot',
        'accuracy': acc,
        'f1_score': f1,
        'auc': 0.0
    }
    
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    print(f"Successfully updated {metrics_file}")

if __name__ == "__main__":
    main()
