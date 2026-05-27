import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class VLMDataset(Dataset):
    def __init__(self, dataframe, image_col, text_col, image_path=None, transform=None):
        self.dataframe = dataframe
        self.image_col = image_col
        self.text_col = text_col
        self.image_path = image_path
        self.transform = transform 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row[self.text_col]
        
        cols_val = row[self.image_col]
        if self.image_path and isinstance(cols_val, str) and not cols_val.startswith('/'):
             image_file = os.path.join(self.image_path, cols_val)
        else:
             image_file = cols_val

        image = Image.open(image_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        return {"text": text, "image": image, "image_path": image_file}

def blip2_collate_fn(batch, processor):
    texts = [f"Question: {item['text']} Answer:" for item in batch]
    images = [item["image"] for item in batch]
    image_paths = [item.get("image_path") for item in batch]
    original_texts = [item["text"] for item in batch]
    
    processed = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)
    
    if all(image_paths):
        processed['image_paths'] = image_paths
        
    processed['original_text'] = original_texts
    return processed

def llava_collate_fn(batch, processor):
    texts = [f"<image>\nUSER: {item['text']}\nASSISTANT:" for item in batch]
    images = [item["image"] for item in batch]
    image_paths = [item.get("image_path") for item in batch]
    original_texts = [item["text"] for item in batch]

    processed = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    
    if all(image_paths):
        processed['image_paths'] = image_paths

    processed['original_text'] = original_texts
    return processed

def clip_collate_fn(batch, processor):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    image_paths = [item.get("image_path") for item in batch]
    original_texts = [item["text"] for item in batch]
    
    processed = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    
    if all(image_paths):
        processed['image_paths'] = image_paths

    processed['original_text'] = original_texts
    return processed

def unimed_collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    image_paths = [item.get("image_path") for item in batch]
    original_texts = [item["text"] for item in batch]
    
    # processor is the transform function
    processed_images = torch.stack([processor(img) for img in images])
    
    batch_out = {"image": processed_images, "text": texts}
    
    if all(image_paths):
        batch_out['image_paths'] = image_paths

    batch_out['original_text'] = original_texts
    return batch_out
