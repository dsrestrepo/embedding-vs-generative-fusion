import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel, AutoModel
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from data_utils import preprocess_df
from vlm_utils import VLMDataset, blip2_collate_fn, llava_collate_fn, clip_collate_fn, unimed_collate_fn
try:
    import open_clip
except ImportError:
    open_clip = None

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Setup distributed computing if available
if torch.distributed.is_available():
    dist.init_process_group(backend="gloo")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


# Functions for getting embeddings
def get_unimed_embeddings(dataframe, batch_size, image_col_name, text_col_name, image_path, output_dir, output_file, processor, model, tokenizer):
    # Initialize the dataset
    dataset = VLMDataset(dataframe, image_col_name, text_col_name, image_path)

    # Set up a DistributedSampler
    sampler = DistributedSampler(dataset, shuffle=False)

    # Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: unimed_collate_fn(x, processor), num_workers=4)

    if dist.get_rank() == 0:
        progress_bar = tqdm(total=len(dataloader), desc="Processing batches")
        
    img_embeddings_list = []
    text_embeddings_list = []
    image_paths_list = []
    original_texts_list = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        image_paths_list.extend(batch['image_paths'])
        if 'original_text' in batch:
            original_texts_list.extend(batch['original_text'])
        else:
            original_texts_list.extend(batch['text'])
            
        # batch['image'] is the tensor of images
        images = batch['image'].to(device)
        texts = batch['text']
        
        # Tokenize text
        text_tokens = tokenizer(texts).to(device)

        with torch.no_grad():
            if hasattr(model, 'module'):
                m = model.module
            else:
                m = model

            # Image Embeddings
            img_emb = m.encode_image(images)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
            
            # Text Embeddings
            text_emb = m.encode_text(text_tokens)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            
        img_embeddings_list.append(img_emb.cpu().numpy())
        text_embeddings_list.append(text_emb.cpu().numpy())
        
        if dist.get_rank() == 0:
            progress_bar.update(1)

    if dist.get_rank() == 0:
        progress_bar.close()

    # Gather
    all_img_embeddings = [None] * dist.get_world_size()
    all_text_embeddings = [None] * dist.get_world_size()
    all_image_paths = [None] * dist.get_world_size()
    all_original_texts = [None] * dist.get_world_size()
    
    dist.all_gather_object(all_img_embeddings, img_embeddings_list)
    dist.all_gather_object(all_text_embeddings, text_embeddings_list)
    dist.all_gather_object(all_image_paths, image_paths_list)
    dist.all_gather_object(all_original_texts, original_texts_list)

    if dist.get_rank() == 0:
        all_image_paths = [item for sublist in all_image_paths for item in sublist]
        all_original_texts = [item for sublist in all_original_texts for item in sublist]
        
        all_img_embeddings = [batch for sublist in all_img_embeddings for batch in sublist]
        img_embeddings = np.concatenate(all_img_embeddings, axis=0)

        all_text_embeddings = [batch for sublist in all_text_embeddings for batch in sublist]
        text_embeddings = np.concatenate(all_text_embeddings, axis=0)

        img_emb_df = pd.DataFrame(img_embeddings, columns=[f'img_emb_{i}' for i in range(img_embeddings.shape[1])])
        text_emb_df = pd.DataFrame(text_embeddings, columns=[f'text_emb_{i}' for i in range(text_embeddings.shape[1])])
        
        # Combine
        embeddings_df = pd.concat([img_emb_df, text_emb_df], axis=1)
        embeddings_df[image_col_name] = all_image_paths
        embeddings_df[text_col_name] = all_original_texts
        
        concatenated_df = pd.merge(dataframe, embeddings_df, on=[image_col_name, text_col_name], how='left')
        output_path = os.path.join(output_dir, output_file)
        concatenated_df.drop_duplicates().to_csv(output_path, index=False)

def get_clip_embeddings(dataframe, batch_size, image_col_name, text_col_name, image_path, output_dir, output_file, processor, model, tokenizer=None):
    # Initialize the dataset
    dataset = VLMDataset(dataframe, image_col_name, text_col_name, image_path)

    # Set up a DistributedSampler
    sampler = DistributedSampler(dataset, shuffle=False)

    # Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: clip_collate_fn(x, processor), num_workers=4)

    if dist.get_rank() == 0:
        progress_bar = tqdm(total=len(dataloader), desc="Processing batches")
        
    img_embeddings_list = []
    text_embeddings_list = []
    image_paths_list = []
    original_texts_list = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_inputs = {k: v.to(device) for k, v in batch.items() if k != 'image_paths' and k != 'original_text'}
        image_paths_list.extend(batch['image_paths'])
        if 'original_text' in batch:
            original_texts_list.extend(batch['original_text'])
        
        with torch.no_grad():
            if hasattr(model, 'module'):
                m = model.module
            else:
                m = model

            # Image Embeddings
            if hasattr(m, 'get_image_features'):
                img_emb = m.get_image_features(pixel_values=batch_inputs['pixel_values'])
            else:
                 # Fallback
                outputs = m(**batch_inputs)
                if hasattr(outputs, 'image_embeds'):
                    img_emb = outputs.image_embeds
                else:
                    img_emb = outputs[0] # Might be risky if output[0] is not image embedding
            
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
            
            # Text Embeddings
            if hasattr(m, 'get_text_features'):
                 text_emb = m.get_text_features(input_ids=batch_inputs['input_ids'], attention_mask=batch_inputs.get('attention_mask'))
            else:
                 # Fallback for models that might return text embeds in forward (unlikely for pure CLIPModel usage here usually separates)
                 # Actually CLIPModel() forward returns CLIPOutput which has text_embeds
                 outputs = m(**batch_inputs)
                 if hasattr(outputs, 'text_embeds'):
                     text_emb = outputs.text_embeds
                 else:
                     text_emb = outputs[1] # Hypothetical fallback

            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

        img_embeddings_list.append(img_emb.cpu().numpy())
        text_embeddings_list.append(text_emb.cpu().numpy())
        
        if dist.get_rank() == 0:
            progress_bar.update(1)

    if dist.get_rank() == 0:
        progress_bar.close()

    # Gather
    all_img_embeddings = [None] * dist.get_world_size()
    all_text_embeddings = [None] * dist.get_world_size()
    all_image_paths = [None] * dist.get_world_size()
    all_original_texts = [None] * dist.get_world_size()
    
    dist.all_gather_object(all_img_embeddings, img_embeddings_list)
    dist.all_gather_object(all_text_embeddings, text_embeddings_list)
    dist.all_gather_object(all_image_paths, image_paths_list)
    dist.all_gather_object(all_original_texts, original_texts_list)

    if dist.get_rank() == 0:
        all_image_paths = [item for sublist in all_image_paths for item in sublist]
        all_original_texts = [item for sublist in all_original_texts for item in sublist]
        
        all_img_embeddings = [batch for sublist in all_img_embeddings for batch in sublist]
        img_embeddings = np.concatenate(all_img_embeddings, axis=0)

        all_text_embeddings = [batch for sublist in all_text_embeddings for batch in sublist]
        text_embeddings = np.concatenate(all_text_embeddings, axis=0)

        img_emb_df = pd.DataFrame(img_embeddings, columns=[f'img_emb_{i}' for i in range(img_embeddings.shape[1])])
        text_emb_df = pd.DataFrame(text_embeddings, columns=[f'text_emb_{i}' for i in range(text_embeddings.shape[1])])
        
        # Combine
        embeddings_df = pd.concat([img_emb_df, text_emb_df], axis=1)
        embeddings_df[image_col_name] = all_image_paths
        embeddings_df[text_col_name] = all_original_texts
        
        concatenated_df = pd.merge(dataframe, embeddings_df, on=[image_col_name, text_col_name], how='left')
        output_path = os.path.join(output_dir, output_file)
        concatenated_df.drop_duplicates().to_csv(output_path, index=False)



# Functions for getting embeddings
def get_blip2_embeddings(dataframe, batch_size, image_col_name, text_col_name, image_path, output_dir, output_file, processor, model, tokenizer=None):
    # Initialize the dataset
    dataset = VLMDataset(dataframe, image_col_name, text_col_name, image_path)

    # Set up a DistributedSampler to partition the dataset among the workers
    sampler = DistributedSampler(dataset, shuffle=False)

    # Initialize the DataLoader with the distributed sampler
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: blip2_collate_fn(x, processor), num_workers=4)

    if dist.get_rank() == 0:
        progress_bar = tqdm(total=len(dataloader), desc="Processing batches")
        
    embeddings_list = []
    image_paths_list = []
    original_texts_list = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        #batch_inputs = {k: v.to(device) for k, v in batch.items() if k != 'image_paths'}
        batch_inputs = {k: v for k, v in batch.items() if k != 'image_paths' and k != 'original_text'}
        image_paths_list.extend(batch['image_paths'])
        if 'original_text' in batch:
            original_texts_list.extend(batch['original_text'])

        with torch.no_grad():
            outputs = model(**batch_inputs)
        
        # Extract embeddings from the model's output
        embeddings = outputs['qformer_outputs']['pooler_output'].cpu().numpy()
        embeddings_list.append(embeddings)
        
        # Update the progress on the master process
        if dist.get_rank() == 0:
            progress_bar.update(1)

    if dist.get_rank() == 0:
        progress_bar.close()

    # Gather all embeddings and image paths from all processes
    all_embeddings = [None] * dist.get_world_size()
    all_image_paths = [None] * dist.get_world_size()
    all_original_texts = [None] * dist.get_world_size()

    dist.all_gather_object(all_embeddings, embeddings_list)
    dist.all_gather_object(all_image_paths, image_paths_list)
    dist.all_gather_object(all_original_texts, original_texts_list)

    # Concatenate embeddings and image paths from all processes
    if dist.get_rank() == 0:  # Only the main process will save the output
        
        all_image_paths = [item for sublist in all_image_paths for item in sublist]
        all_original_texts = [item for sublist in all_original_texts for item in sublist]
        all_embeddings = [batch for sublist in all_embeddings for batch in sublist]
        
        embeddings = np.concatenate(all_embeddings, axis=0)

        embeddings_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])
        embeddings_df[image_col_name] = all_image_paths
        embeddings_df[text_col_name] = all_original_texts

        # Merge the original dataframe with the embeddings dataframe on the image path
        concatenated_df = pd.merge(dataframe, embeddings_df, on=[image_col_name, text_col_name], how='left')

        # Save the final DataFrame to a CSV file
        output_path = os.path.join(output_dir, output_file)
        concatenated_df.drop_duplicates().to_csv(output_path, index=False)


def get_llava_embeddings(dataframe, batch_size, image_col_name, text_col_name, image_path, output_dir, output_file, processor, model, tokenizer=None):
   # Initialize the dataset
    dataset = VLMDataset(dataframe, image_col_name, text_col_name, image_path)
    
    # Set up a DistributedSampler to partition the dataset among the workers
    sampler = DistributedSampler(dataset, shuffle=False)

    # Initialize the DataLoader with the distributed sampler
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: llava_collate_fn(x, processor), num_workers=4)
    
    if dist.get_rank() == 0:
        progress_bar = tqdm(total=len(dataloader), desc="Processing batches")

    embeddings_list = []
    image_paths_list = []
    original_texts_list = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        #batch_inputs = {k: v.to(device) for k, v in batch.items() if k != 'image_paths'}
        batch_inputs = {k: v for k, v in batch.items() if k != 'image_paths' and k != 'original_text'}
        image_paths_list.extend(batch['image_paths'])
        if 'original_text' in batch:
            original_texts_list.extend(batch['original_text'])
        
        with torch.no_grad():
            outputs = model(**batch_inputs, output_hidden_states=True)
            
        
        # Extract embeddings from the last hidden state and perform mean pooling with attention mask
        last_hidden_state = outputs.hidden_states[-1]  # Shape: (batch_size, sequence_length, hidden_size)
        
        if 'attention_mask' in batch_inputs:
            input_mask_expanded = batch_inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        else:
            pooled_embeddings = last_hidden_state.mean(dim=1).cpu().numpy()  # Shape: (batch_size, hidden_size)
        
        embeddings_list.append(pooled_embeddings)

    # Gather all embeddings and image paths from all processes
    all_embeddings = [None] * dist.get_world_size()
    all_image_paths = [None] * dist.get_world_size()
    all_original_texts = [None] * dist.get_world_size()
    dist.all_gather_object(all_embeddings, embeddings_list)
    dist.all_gather_object(all_image_paths, image_paths_list)
    dist.all_gather_object(all_original_texts, original_texts_list)

    # Concatenate embeddings and image paths from all processes
    if dist.get_rank() == 0:  # Only the main process will save the output
        
        all_image_paths = [item for sublist in all_image_paths for item in sublist]
        all_original_texts = [item for sublist in all_original_texts for item in sublist]
        
        all_embeddings = [batch for sublist in all_embeddings for batch in sublist]

        
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        print(embeddings.shape)
        print(len(all_image_paths))

        embeddings_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])
        embeddings_df[image_col_name] = all_image_paths
        embeddings_df[text_col_name] = all_original_texts

        # Merge the original dataframe with the embeddings dataframe on the image path
        concatenated_df = pd.merge(dataframe, embeddings_df, on=[image_col_name, text_col_name], how='left')

        # Save the final DataFrame to a CSV file
        output_path = os.path.join(output_dir, output_file)
        concatenated_df.drop_duplicates().to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from VLMs.")
    parser.add_argument("--classifier", type=str, required=True, help="Classifier to use: LLAVA or BLIP2")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--image_col", type=str, required=True, help="Name of the image column")
    parser.add_argument("--text_col", type=str, required=True, help="Name of the text column")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for embeddings")
    parser.add_argument("--output_file", type=str, required=True, help="Output file name for embeddings")
    parser.add_argument("--image_dir", type=str, required=False, default='images', help="Name of directory with the images to the dataset")    
    parser.add_argument("--labels", type=str, required=False, default='labels.csv', help="Name of the file with the labels ans text")
    args = parser.parse_args()

    print('Starting embedding extraction with the following parameters:')
    print(f"Classifier: {args.classifier}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Output File: {args.output_file}")




    # Load dataframe
    labels_path = os.path.join(args.dataset_path, args.labels)
    images_path = os.path.join(args.dataset_path, args.image_dir)
    
    if 'coco' in args.dataset_path.lower():
        df = preprocess_df(df=pd.read_csv(labels_path), image_columns=args.image_col, images_path=images_path, coco_format=True)
    else:
        df = preprocess_df(df=pd.read_csv(labels_path), image_columns=args.image_col, images_path=images_path)
    
    # Filter out rows with missing text
    if args.text_col in df.columns:
        initial_len = len(df)
        df = df[df[args.text_col].notna()]
        df = df[df[args.text_col].astype(str).str.strip() != ""]
        if len(df) < initial_len:
            print(f"Dropped {initial_len - len(df)} rows with missing or empty text in column '{args.text_col}'.")

    os.makedirs(args.output_dir, exist_ok=True)


    # Initialize model and processor
    tokenizer = None
    if args.classifier.lower() == 'blip2':
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")#.to(device)
        
        if torch.distributed.is_available() and not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = torch.nn.parallel.DistributedDataParallel(model)#, device_ids=[local_rank], output_device=local_rank)

        get_embeddings = get_blip2_embeddings
    elif args.classifier.lower() == 'llava':
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")#.to(device)
        
        if torch.distributed.is_available() and not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = torch.nn.parallel.DistributedDataParallel(model)#, device_ids=[local_rank], output_device=local_rank)
        get_embeddings = get_llava_embeddings

    elif args.classifier.lower() == 'clip':
        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device)
        if torch.distributed.is_available():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        get_embeddings = get_clip_embeddings

    elif args.classifier.lower() == 'siglip':
        model_id = "google/siglip-base-patch16-384"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        if torch.distributed.is_available():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        get_embeddings = get_clip_embeddings

    elif args.classifier.lower() == 'medsiglip':
        model_id = "google/medsiglip-448"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        if torch.distributed.is_available():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        get_embeddings = get_clip_embeddings

    elif args.classifier.lower() == 'unimed':
        if open_clip is None:
            raise ImportError("open_clip is not installed. Please install it.")
        
        # Define model parameters
        model_name = 'ViT-L-14-336-quickgelu' #'ViT-B-16-quickgelu' # 'ViT-L-14-336-quickgelu'
        
        # print("warning: USING UNIMED MODEL WITH OPENCLIP - WEIGHTS MUST BE DOWNLOADED TO /gpfs/workdir/restrepoda/models/unimed_clip_vit_b16.pt")
        #pretrained = "/gpfs/workdir/restrepoda/models/unimed_clip_vit_b16.pt"
        
        # print("warning: USING UNIMED MODEL WITH OPENCLIP - WEIGHTS MUST BE DOWNLOADED TO /gpfs/workdir/restrepoda/models/unimed_clip_vit_l14_base_text_encoder.pt")
        pretrained = "/gpfs/workdir/restrepoda/models/unimed_clip_vit_l14_base_text_encoder.pt"

        text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        
        # Check if pretrained weights exist, otherwise warn or fail
        if not os.path.exists(pretrained):
            # If not in current dir, check if provided in args? For now assume it logic is imperative
            print(f"Warning: {pretrained} not found. Please ensure the weights are downloaded to this path.")
        
        #mean, std = open_clip.get_mean_std()

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
            #weights_only=False,
            force_quick_gelu=True,
            #mean=mean, std=std,
            pretrained_image=False,
            inmem=True,
            text_encoder_name=text_encoder_name
        )
        processor = preprocess 
        # Get tokenizer for UniMed
        tokenizer = open_clip.HFTokenizer(
            text_encoder_name,
            context_length=256
        )
        
        if torch.distributed.is_available():
            # open_clip models are torch modules
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            
        get_embeddings = get_unimed_embeddings

    elif args.classifier.lower() == 'biomedclip':
        if open_clip is None:
            raise ImportError("open_clip is not installed. Please install it.")
        
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        model, preprocess = open_clip.create_model_from_pretrained(model_name)
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer(model_name)
        processor = preprocess

        if torch.distributed.is_available():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        
        get_embeddings = get_unimed_embeddings

    else:
        raise ValueError("Unsupported classifier. Choose either LLAVA, BLIP2, CLIP, SIGLIP, MEDSIGLIP, UNIMED, or BIOMEDCLIP.")


    # Get embeddings
    print('Starting...')
    get_embeddings(df, args.batch_size, args.image_col, args.text_col, images_path, args.output_dir, args.output_file, processor, model, tokenizer=tokenizer)

if __name__ == "__main__":
    main()
