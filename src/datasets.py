from torch.utils.data import Dataset
import torch
import numpy as np

# Custom Dataset class for PyTorch
class VQADataset(Dataset):
    """
    Custom PyTorch Dataset for VQA (Visual Question Answering).

    Args:
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - text_cols (list): List of column names containing text data.
    - image_cols (list): List of column names containing image data.
    - label_col (str): Column name containing labels (unused in __getitem__ directly if labels are passed separately, but init logic suggests labels might be handled here).
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.
    - labels (np.ndarray): Optional, if labels are pre-processed and passed.
    """
    def __init__(self, df, text_cols, image_cols, label_col, mlb, train_columns, labels=None):
        # Pre-convert to tensors for efficiency
        self.text_data = torch.tensor(df[text_cols].values, dtype=torch.float32)
        self.image_data = torch.tensor(df[image_cols].values, dtype=torch.float32)
        self.mlb = mlb
        self.train_columns = train_columns
        
        if labels is not None:
            # Handle if labels is a Series or DataFrame or numpy array
            vals = labels.values if hasattr(labels, 'values') else labels
            self.labels = torch.tensor(vals, dtype=torch.float32)
        elif label_col in df.columns:
            encoded = self.mlb.transform(df[label_col])
            self.labels = torch.tensor(encoded, dtype=torch.float32)
        else:
             self.labels = None

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        # Direct indexing of tensors is faster
        text_sample = self.text_data[idx]
        image_sample = self.image_data[idx]
        
        if self.labels is not None:
            label = self.labels[idx]
            return {'text': text_sample, 'image': image_sample, 'labels': label}
        else:
            return {'text': text_sample, 'image': image_sample}
