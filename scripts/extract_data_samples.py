import os
import pandas as pd
import shutil
import sys
import argparse

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from data_utils import preprocess_df
except ImportError:
    print("Could not import preprocess_df from src/data_utils.py. Make sure you are running from the project root.")
    sys.exit(1)

# Dataset configurations
DATASET_CONFIGS = {
    'daquar': {'path': 'daquar', 'image_col': 'image_id', 'text_col': 'question', 'image_dir': 'images', 'label_col': 'answer'},
    'coco-qa': {'path': 'coco-qa', 'image_col': 'image_id', 'text_col': 'questions', 'image_dir': 'images', 'label_col': 'answers'},
    'fakeddit': {'path': 'fakeddit', 'image_col': 'id', 'text_col': 'text', 'image_dir': 'images', 'label_col': '2_way_label'},
    'Recipes5k': {'path': 'Recipes5k', 'image_col': 'image', 'text_col': 'ingredients', 'image_dir': 'images', 'label_col': 'class'},
    'brset': {'path': 'BRSET/brset', 'image_col': 'image_id', 'text_col': 'text', 'image_dir': 'images', 'label_col': 'DR_2'},
    'ham10000': {'path': 'HAM10000', 'image_col': 'image_id', 'text_col': 'text', 'image_dir': 'images', 'label_col': 'dx'},
    'mimic': {'path': 'MIMIC/mimic', 'image_col': 'path_preproc', 'text_col': 'text', 'image_dir': '.', 'label_col': 'disease_label'},
    'mbrset': {'path': 'mBRSET/mbrset', 'image_col': 'file', 'text_col': 'text', 'image_dir': 'images', 'label_col': 'DR_2'},
}

def main():
    parser = argparse.ArgumentParser(description="Extract sample data from datasets.")
    parser.add_argument("--dataset_root", type=str, default='/gpfs/workdir/restrepoda/datasets', help="Root directory of datasets")
    parser.add_argument("--output_root", type=str, default='data_samples', help="Output directory for samples")
    args = parser.parse_args()

    DATASET_ROOT = args.dataset_root
    OUTPUT_ROOT = args.output_root

    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    for dataset_name in DATASET_CONFIGS:
        print(f"Processing {dataset_name}...")
        config = DATASET_CONFIGS[dataset_name]
        dataset_path = os.path.join(DATASET_ROOT, config['path'])
        
        # Check if dataset path exists
        if not os.path.exists(dataset_path):
             print(f"  Skipping {dataset_name}: Path {dataset_path} not found.")
             continue
        
        # Determine labels file
        csv_path = os.path.join(dataset_path, 'labels.csv')
        if not os.path.exists(csv_path):
            print(f"  labels.csv not found in {dataset_path}")
            continue

        try:
            # Determine images path
            images_dir = os.path.join(dataset_path, config['image_dir'])
            
            df = pd.read_csv(csv_path)
            
            # Special handling for mimic
            is_mimic = 'mimic' in dataset_name.lower()
            disease_cols = [
                "Atelectasis", 
                "Cardiomegaly", 
                "Consolidation", 
                "Edema",
                "No Finding", 
                "Pleural Effusion"
            ]
            
            # Use preprocess_df to fix image paths
            image_col = config['image_col']
            
            # Special handling for coco-qa format in preprocess_df
            is_coco = 'coco' in dataset_name.lower()
            
            # preprocess_df returns dataframe with VALID images only and corrected paths
            # It expects images_path to be the directory containing images
            
            # Note: preprocess_df modifies dataframe inplace but returns it too?
            # It returns filtered df.
            # However, preprocess_df signature: preprocess_df(df, image_columns, images_path, coco_format=False)
            
            # Need to be careful. preprocess_df calls Parallel which might fail if not main guarded on some OS
            # But we are in main()
            
            # Only process a subset to speed up valid check if dataset is huge?
            # But we need valid images.
            # Let's take a sample FIRST, then check validity.
            
            df_sample = df.sample(min(20, len(df)), random_state=42) # Take 20 candidates
            
            # We must fix paths for these 20
            # preprocess_df processes the WHOLE dataframe. This might be slow if we just want 2 samples.
            # Let's manually do what preprocess_df does for just these 20 samples.
            
            valid_samples = []
            count = 0
            TARGET_SAMPLES = 10
            
            for index, row in df_sample.iterrows():
                if count >= TARGET_SAMPLES: break
                
                img_val = row[image_col]
                
                # Logic from preprocess_df: correct_image_path
                full_img_path = str(img_val)
                if not os.path.isabs(full_img_path):
                     full_img_path = os.path.join(images_dir, full_img_path)
                
                # Check extensions
                if not os.path.exists(full_img_path):
                    # Try appending jpg
                    if os.path.exists(full_img_path + '.jpg'):
                        full_img_path += '.jpg'
                    elif os.path.exists(full_img_path + '.png'):
                        full_img_path += '.png'
                    elif os.path.exists(full_img_path + '.tiff'):
                        full_img_path += '.tiff'
                    else:
                        # Try to match filename in dir (slow)
                        dirname = os.path.dirname(full_img_path)
                        basename = os.path.basename(full_img_path)
                        if os.path.exists(dirname):
                            found = False
                            for f in os.listdir(dirname):
                                if f.startswith(basename) and f != basename:
                                    full_img_path = os.path.join(dirname, f)
                                    found = True
                                    break
                            if not found:
                                print(f"  Image not found: {full_img_path}")
                                continue
                        else:
                             print(f"  Dir not found: {dirname}")
                             continue

                # Found valid image
                valid_samples.append((row, full_img_path))
                count += 1
            
            if not valid_samples:
                print(f"  No valid samples found in 20 candidates.")
                continue

            # Create output dir
            ds_out_dir = os.path.join(OUTPUT_ROOT, dataset_name)
            if not os.path.exists(ds_out_dir):
                os.makedirs(ds_out_dir)

            for i, (row, src_image_path) in enumerate(valid_samples):
                dataset_idx = i + 1
                
                target_ext = os.path.splitext(src_image_path)[1]
                if not target_ext: target_ext = ".jpg" # Default
                
                dst_image_name = f"example_image{dataset_idx}{target_ext}"
                dst_image_path = os.path.join(ds_out_dir, dst_image_name)
                
                shutil.copy(src_image_path, dst_image_path)
                
                # Text
                text_col = config['text_col']
                text = row[text_col]
                with open(os.path.join(ds_out_dir, f"text{dataset_idx}.txt"), "w") as f:
                    f.write(str(text))
                
                # Label
                label_col = config['label_col']
                
                if is_mimic:
                    # Construct multi-label string
                    active_diseases = []
                    for disease in disease_cols:
                        if disease in row and row[disease] == 1:
                            active_diseases.append(disease)
                    label = ", ".join(active_diseases) if active_diseases else "No Finding"
                else:
                    label = row[label_col] if label_col in df.columns else "N/A"
                    
                with open(os.path.join(ds_out_dir, f"label{dataset_idx}.txt"), "w") as f:
                    f.write(str(label))
                
            print(f"  Saved {len(valid_samples)} samples to {ds_out_dir}")

        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
