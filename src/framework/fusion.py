"""Framework wrapper for frozen-embedding fusion experiments."""
import pandas as pd

from scripts.fusion_models import train_and_predict


class FusionTrainerFramework:
    def __init__(self, embeddings_path, label_col, text_prefix="text_emb_", image_prefix="img_emb_",
                 num_samples=None, is_multilabel=False, dataset_name=""):
        if is_multilabel:
            raise NotImplementedError("The rebuttal runner currently supports the paper's single-label tasks.")
        self.df = pd.read_csv(embeddings_path)
        self.label_col = label_col[0] if isinstance(label_col, list) else label_col
        self.dataset_name = dataset_name
        self.text_cols = [c for c in self.df.columns if c.startswith(text_prefix)]
        self.img_cols = [c for c in self.df.columns if c.startswith(image_prefix)]
        if not self.text_cols or not self.img_cols:
            raise ValueError("Could not find text_emb_ and img_emb_ columns.")
        self.df = self.df.dropna(subset=[self.label_col] + self.text_cols + self.img_cols).copy()
        if num_samples:
            self.df = self.df.sample(min(num_samples, len(self.df)), random_state=42).copy()

    def run_method(self, method, random_state=42, num_epochs=100, batch_size=4096, lr=3e-4, patience=5):
        metrics, test_df, y_true, probabilities, gates, model = train_and_predict(
            self.df, self.text_cols, self.img_cols, self.label_col, method, seed=random_state,
            epochs=num_epochs, batch_size=batch_size, lr=lr, patience=patience,
        )
        metrics["model"] = method
        return metrics, test_df, y_true, probabilities, gates, model

    def run_early_fusion_baseline(self):
        return self.run_method("linear")[0]

    def run_mlp_fusion(self, fusion_type="early", num_epochs=100):
        return self.run_method(f"mlp_{fusion_type}", num_epochs=num_epochs)[0]
