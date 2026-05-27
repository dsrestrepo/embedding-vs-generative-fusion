import pandas as pd
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.data_utils import split_data, process_labels
from src.datasets import VQADataset
from src.classifiers import train_early_fusion, train_late_fusion

class FusionTrainerFramework:
    """
    A unified PyTorch framework for training Early, Late, and Linear Probes 
    on pre-extracted Multimodal Embeddings using distributed-ready Loaders.
    """
    def __init__(self, embeddings_path: str, label_col: list | str, text_prefix: str = "text_emb_", image_prefix: str = "img_emb_", num_samples: int = None, is_multilabel: bool = False, dataset_name: str = ""):
        print(f"Loading generic fusion framework on {embeddings_path}...")
        self.df = pd.read_csv(embeddings_path)
        
        if isinstance(label_col, str):
            self.label_col = [label_col]
        else:
            self.label_col = label_col
            
        self.is_multilabel = is_multilabel
        self.dataset_name = dataset_name
        
        self.text_cols = [c for c in self.df.columns if text_prefix in c]
        self.img_cols = [c for c in self.df.columns if image_prefix in c]
        
        # Special handling for Mimic multilabel
        if self.dataset_name.lower() == 'mimic' and self.is_multilabel:
            self.df = self.df[(self.df[self.label_col] != -1).all(axis=1)]
            self.df[self.label_col] = self.df[self.label_col].fillna(0)
        
        # Ensure labels exist and clean NaNs (if target varies dynamically, we dropna based on it)
        self.df = self.df.dropna(subset=self.label_col + self.text_cols + self.img_cols)
        
        # Subsample if requested (user said "use full dataset", so num_samples is None by default)
        if num_samples and num_samples > 0 and num_samples < len(self.df):
            print(f"Subsampling dataset to {num_samples} samples, attempting to balance classes...")
            class_counts = self.df[self.label_col].value_counts()
            n_classes = len(class_counts)
            
            sampled_dfs = []
            remaining = num_samples
            
            for c, count in class_counts.sort_values().items():
                n_samp = min(count, remaining // n_classes) if n_classes > 0 else remaining
                sampled_dfs.append(self.df[self.df[self.label_col] == c].sample(n=n_samp, random_state=42))
                remaining -= n_samp
                n_classes -= 1
                
            self.df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    def run_training(self, fusion_type="early", random_state=42, num_epochs=10, batch_size=4096, lr=3e-4, hidden=[128], patience=5):
        """
        Trains a PyTorch model: Early Fusion, Late Fusion, or Linear Probe (if hidden=0).
        Utilizes `loaders` from VQADataset supporting hardware acceleration and distributed torch parameters natively.
        """
        print(f"\nRunning {fusion_type.capitalize()} Fusion with hidden layer size(s) {hidden} on {len(self.df)} samples...")
        
        # 1. Native Train/Test splits using `split_data` (leverages "split" column if it exists)
        splits = split_data(self.df, val_size=0.1, random_state=random_state)
        
        if len(splits) == 3:
            train_df, val_df, test_df = splits
        else:
            train_df, test_df = splits
            val_df = None

        print(f"Initial Train/Val/Test sizes -> Train: {len(train_df)}, Val: {len(val_df) if val_df is not None else 0}, Test: {len(test_df)}")

        # --- DAQUAR Specific Filter since it has too many classes
        if self.dataset_name.lower() == 'daquar':
            print(f"Applying {self.dataset_name} filter: keeping classes with >= 20 samples in test set.")
            label_col_arg = self.label_col[0] if len(self.label_col) == 1 else self.label_col
            
            if label_col_arg in test_df.columns:
                temp_test_series = test_df[label_col_arg]
                
                # Check if values are lists (or string rep of lists)
                first_val = temp_test_series.iloc[0] if len(temp_test_series) > 0 else ""
                if isinstance(first_val, str) and ',' in first_val:
                    temp_test_series = temp_test_series.str.split(',').explode().str.strip()
                
                label_counts = temp_test_series.value_counts()
                valid_classes = label_counts[label_counts >= 20].index.tolist()
                
                print(f"Found {len(valid_classes)} valid classes with >= 20 samples in test set.")
                
                def filter_rows(row, valid):
                    val = row[label_col_arg]
                    if isinstance(val, str):
                        parts = [p.strip() for p in val.split(',')]
                        return any(p in valid for p in parts)
                    return val in valid

                train_df = train_df[train_df.apply(lambda row: filter_rows(row, valid_classes), axis=1)]
                if val_df is not None:
                    val_df = val_df[val_df.apply(lambda row: filter_rows(row, valid_classes), axis=1)]
                test_df = test_df[test_df.apply(lambda row: filter_rows(row, valid_classes), axis=1)]
                
                print(f"Filtered sizes - Train: {len(train_df)}, Val: {len(val_df) if val_df is not None else 0}, Test: {len(test_df)}")
        # ------------------------------ END DAQUAR Filter ------------------------------

        # 2. Process Labels via Data Utils
        train_labels, mlb, train_target_columns = process_labels(train_df, col=(self.label_col[0] if len(self.label_col) == 1 else self.label_col))
        if val_df is not None:
             val_labels = process_labels(val_df, col=(self.label_col[0] if len(self.label_col) == 1 else self.label_col), train_columns=train_target_columns, mlb=mlb)
        test_labels = process_labels(test_df, col=(self.label_col[0] if len(self.label_col) == 1 else self.label_col), train_columns=train_target_columns, mlb=mlb)

        # 3. Create High-Efficiency Datasets (Reusing imported VQADataset)
        train_dataset = VQADataset(train_df, self.text_cols, self.img_cols, (self.label_col[0] if len(self.label_col) == 1 else self.label_col), mlb, train_target_columns, labels=train_labels)
        if val_df is not None:
             val_dataset = VQADataset(val_df, self.text_cols, self.img_cols, (self.label_col[0] if len(self.label_col) == 1 else self.label_col), mlb, train_target_columns, labels=val_labels)
        test_dataset = VQADataset(test_df, self.text_cols, self.img_cols, (self.label_col[0] if len(self.label_col) == 1 else self.label_col), mlb, train_target_columns, labels=test_labels)

        # 4. Standard Dataloaders (with pin_memory for GPUs)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
        if val_df is not None:
             val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
        else:
             val_loader = None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

        # Output resolution logic
        if hasattr(train_dataset, 'labels'):
            output_size = train_dataset.labels.shape[1] if len(train_dataset.labels.shape) > 1 else 1
        else:
            output_size = 1

        # 5. Model execution hyperparameters
        kwargs = {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "text_input_size": len(self.text_cols),
            "image_input_size": len(self.img_cols),
            "output_size": output_size,
            "num_epochs": num_epochs,
            "multilabel": self.is_multilabel,
            "report": False,
            "V": False,
            "lr": lr,
            "set_weights": True,
            "adam": True,
            "p": 0.2, # Dropout prob
            "val_loader": val_loader if val_loader else test_loader,
            "patience": patience,
            "hidden": hidden
        }
        
        if fusion_type.lower() == "early":
            accuracy, precision, recall, f1, best = train_early_fusion(**kwargs)
        elif fusion_type.lower() == "late":
            accuracy, precision, recall, f1, best = train_late_fusion(**kwargs)
        else:
            raise ValueError("fusion_type must be 'early' or 'late'")
            
        print("MLP Training Complete. Extracting best metrics...")
        
        # Grab best metrics from Macro-F1 profile if available
        metrics = best.get('Macro-F1', best.get('Acc', {}))
        
        return {
            "model": f"Torch_{fusion_type.capitalize()}_Fusion_H{hidden}",
            "accuracy": metrics.get('Acc', accuracy),
            "f1_score": metrics.get('F1', f1),
            "auc": metrics.get('Auc', 0),
            "epochs_run": metrics.get('Epoch', num_epochs)
        }

    # Shim methods for legacy API backwards compatibility on the runner jobs
    def run_early_fusion_baseline(self):
        # Hidden=0 explicitly forces Linear Probe logic in classifier
        return self.run_training(fusion_type="early", hidden=0, num_epochs=100)

    def run_mlp_fusion(self, fusion_type="early", num_epochs=100):
        # Standard MLP topology inside early/late
        return self.run_training(fusion_type=fusion_type, hidden=[128], num_epochs=num_epochs)
