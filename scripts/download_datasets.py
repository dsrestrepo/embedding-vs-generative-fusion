import os
import argparse
import pandas as pd
import zipfile
import sys
from typing import Any, Dict, List

import yaml

# Ensure src can be imported
sys.path.append(os.getcwd())

from src.get_data import (
    get_daquar_dataset, preprocess_daquar_dataset,
    get_cocoqa_dataset, process_cocoqa_data,
    download_fakeddit_files, create_stratified_subset_fakeddit, download_full_set_images, download_images_from_file,
    process_fakeddit_dataset,
    download_recipes5k_dataset, preprocess_recipes5k,
    get_brset, brset_preprocessing,
    get_satellitedata, satellitedata_preprocessing,
    preprocess_ham10000
)
from sklearn.model_selection import train_test_split

def process_daquar(output_dir):
    print("Processing DAQUAR...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    get_daquar_dataset(output_dir)
    preprocess_daquar_dataset(output_dir)

def process_cocoqa(output_dir):
    print("Processing COCO-QA...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    get_cocoqa_dataset(output_dir)
    process_cocoqa_data(output_dir)

def process_fakeddit(output_dir, subset_size=1.0, download_mode="url"):
    print(f"Processing Fakeddit (subset size: {subset_size}, mode: {download_mode})...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Download TSV files
    download_fakeddit_files(output_dir)
    
    if download_mode == "drive":
        # Download the full 100GB images tarball from Drive
        print("Downloading full dataset from Google Drive...")
        download_full_set_images(output_dir)
    
    # Create the stratified subset CSV (labels.csv)
    # This will generate the list of images we actually need
    create_stratified_subset_fakeddit(output_dir, subset_size)
    
    if download_mode == "url":
        # Download only the specific images for the subset from their source URLs
        print("Downloading images from URLs (this avoids downloading the full 100GB dataset)...")
        download_images_from_file(output_dir)
        print('Done! Since some images can still be missing due to URL issues, or can be corrupted, we recommend to check the notebook in Notebooks/check_fakeddit_images.ipynb to verify the integrity of the downloaded images and handle any missing/corrupted files.')

    
def process_recipes5k(output_dir):
    print("Processing Recipes5k...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    download_recipes5k_dataset(output_dir)
    # The function generates 'Recipes5k' content but preprocess likely expects the full path
    preprocess_recipes5k(os.path.join(output_dir, 'Recipes5k'))

def process_brset(output_dir):
    print("Processing BRSET...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    get_brset(output_dir, download=True)
    brset_preprocessing(output_dir)

def process_satellite(output_dir):
    print("Processing Satellite Data...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    get_satellitedata(output_dir, download=False)
    num_classes = 3
    df = satellitedata_preprocessing(output_dir, num_classes=num_classes)
    
    # Fix image path
    df.image_id = df.image_id.apply(lambda x: x.split('/')[-1].replace('image', x.split('/')[-2]).replace('.tiff', '.jpg'))
    
    cities =  {
        "76001": "Cali",
        "5001": "Medellín",
        "50001": "Villavicencio",
        "54001": "Cúcuta",
        "73001": "Ibagué",
        "68001": "Bucaramanga",
        "5360": "Itagüí",
        "8001": "Barranquilla",
        "41001": "Neiva",
        "23001": "Montería"
    }

    df['text'] = df.apply(lambda x: f"An image from city {cities[x.image_id.split('_')[0]]} taken in date {x.image_id.split('_')[1].replace('.tiff', '')} with"+x.text[x.text.index(','):], axis=1)
    df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)

def process_mimic(mimic_path):
    print("Processing MIMIC...")
    
    train = pd.read_csv(os.path.join(mimic_path, 'train.csv'), index_col=0)
    test = pd.read_csv(os.path.join(mimic_path, 'test.csv'), index_col=0)
    val = pd.read_csv(os.path.join(mimic_path, 'valid.csv'), index_col=0)

    train['split'] = 'train'
    test['split'] = 'test'
    val['split'] = 'val'

    df = pd.concat([train, test, val], ignore_index=True)

    text_path = pd.read_csv(os.path.join(mimic_path, 'cxr-study-list.csv'))
    text_path.rename(columns={'path': 'file_path'}, inplace=True)

    df = pd.merge(df, text_path)

    zip_file_path = os.path.join(mimic_path, 'metadata', 'mimic-cxr-reports.zip')
    
    def extract_texts_from_zip(zip_path, file_list):
        file_contents = {}
        with zipfile.ZipFile(zip_path, 'r') as z:
            existing_files = set(z.namelist()) 
            for file_path in file_list:
                if file_path in existing_files:
                    with z.open(file_path) as f:
                        file_contents[file_path] = f.read().decode('utf-8')
                else:
                    file_contents[file_path] = None 
        return file_contents

    texts = extract_texts_from_zip(zip_file_path, df['file_path'].tolist())
    df['text'] = df['file_path'].map(texts)
    df.to_csv(os.path.join(mimic_path, 'labels.csv'), index=False)

def process_mbrset(dataset_path):
    print("Processing mBRSET...")
    # Helper code from notebook
    def generate_patient_text(row):
        def binary_to_text(value, true_text, false_text):
            return true_text if value == 1 else false_text
        
        education_map = {
            1.0: "illiterate",
            2.0: "with incomplete primary education",
            3.0: "with complete primary education",
            4.0: "with incomplete secondary education",
            5.0: "with complete secondary education",
            6.0: "with incomplete tertiary education",
            7.0: "with complete tertiary education"
        }
        
        age_description = f"aged {row['age']} years" if not pd.isnull(row['age']) else "with age not reported"
        sex_description = "male" if row['sex'] == 1 else "female" if row['sex'] == 0 else "sex not reported"
        dm_duration = f"diagnosed with diabetes for {row['dm_time']} years" if not pd.isnull(row['dm_time']) else "with no reported diabetes duration"
        
        insulin_use = binary_to_text(row['insulin'], "using insulin", "not using insulin")
        oral_treatment = binary_to_text(row['oraltreatment_dm'], "on oral treatment for diabetes", "not on oral treatment for diabetes")
        hypertension = binary_to_text(row['systemic_hypertension'], "with systemic hypertension", "without systemic hypertension")
        alcohol_use = binary_to_text(row['alcohol_consumption'], "consumes alcohol", "does not consume alcohol")
        smoking = binary_to_text(row['smoking'], "smokes", "does not smoke")
        obesity = binary_to_text(row['obesity'], "with obesity", "without obesity")
        vascular_disease = binary_to_text(row['vascular_disease'], "has vascular disease", "does not have vascular disease")
        myocardial_infarction = binary_to_text(row['acute_myocardial_infarction'], "has a history of acute myocardial infarction", "no history of acute myocardial infarction")
        nephropathy = binary_to_text(row['nephropathy'], "with nephropathy", "without nephropathy")
        neuropathy = binary_to_text(row['neuropathy'], "with neuropathy", "without neuropathy")
        diabetic_foot = binary_to_text(row['diabetic_foot'], "has diabetic foot", "does not have diabetic foot")
        education_description = education_map.get(row['educational_level'], "with no educational level reported")
        
        description = (
            f"A {sex_description} patient {age_description}, {dm_duration}, {insulin_use}, and {oral_treatment}. "
            f"The patient is {hypertension}, {alcohol_use}, {smoking}, {obesity}, and {vascular_disease}. "
            f"Medical history includes: {myocardial_infarction}, {nephropathy}, {neuropathy}, and {diabetic_foot}. "
            f"The patient is {education_description}."
        )
        return description

    filename = 'labels_mbrset.csv'
    output_filename = 'labels.csv'
    
    df = pd.read_csv(os.path.join(dataset_path, filename))
    df['text'] = df.apply(generate_patient_text, axis=1)

    df.rename(columns={'final_icdr': 'DR_ICDR'}, inplace=True)
    df = df[['file', 'DR_ICDR', 'text']]
    
    df.dropna(subset = ['DR_ICDR'], inplace=True)

    df['DR_2'] = df['DR_ICDR'].apply(lambda x: 1 if x > 0 else 0)
    df['DR_3'] = df['DR_ICDR'].apply(lambda x: 2 if x == 4 else (1 if x in [1, 2, 3] else 0))

    df['split'] = 'train'
    try:
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df['DR_ICDR'], random_state=42)
        df.loc[test_idx, 'split'] = 'test'
    except ValueError as e:
        print(f"Warning: Could not stratify split: {e}")

    df.to_csv(os.path.join(dataset_path, output_filename), index=False)
    print(f"Processed dataset saved as {output_filename} in {dataset_path}")

def process_ham10000(output_dir):
    print("Processing HAM10000...")
    print("WARNING: HAM10000 dataset should already be downloaded and placed in the directory. This function will only preprocess it.")
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist. Please download the dataset first.")
        return
    preprocess_ham10000(output_dir)

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def enabled_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in items if item.get("enabled", False)]


def build_parser():
    parser = argparse.ArgumentParser(description="Download and preprocess datasets.")
    parser.add_argument("--config", type=str, help="Run enabled download_datasets entries from a YAML config")
    parser.add_argument("--dataset", type=str, help="Specific dataset to process (daquar, coco-qa, fakeddit, recipes5k, brset, satellite, mimic, mbrset, or all)", default="all")
    parser.add_argument("--fakeddit_subset", type=float, help="Subset fraction for Fakeddit (e.g. 0.2 for 20%)", default=1.0)
    parser.add_argument("--fakeddit_download_mode", type=str, choices=["url", "drive"], default="url", help="Download mode for Fakeddit images: 'url' (individual images from web) or 'drive' (full tarball from Google Drive)")
    parser.add_argument("--data_dir", type=str, help="Base directory to save datasets", default="datasets/")
    return parser


def run_download(args):
    base_path = args.data_dir

    if args.dataset in ["daquar", "all"]:
        process_daquar(os.path.join(base_path, "daquar"))
        
    if args.dataset in ["coco-qa", "all"]:
        process_cocoqa(os.path.join(base_path, "coco-qa"))

    if args.dataset in ["fakeddit", "all"]:
        process_fakeddit(os.path.join(base_path, "fakeddit"), args.fakeddit_subset, args.fakeddit_download_mode)

    if args.dataset in ["recipes5k", "all"]:
        # Recipes5k function expects the parent folder as it extracts a 'Recipes5k' folder
        process_recipes5k(base_path)

    if args.dataset in ["brset", "all"]:
        process_brset(os.path.join(base_path, "brset"))

    if args.dataset in ["satellite", "all"]:
        process_satellite(os.path.join(base_path, "satellitedata"))

    if args.dataset in ["mimic", "all"]:
        # MIMIC has a nested structure in the notebook
        process_mimic(os.path.join(base_path, "MIMIC/mimic"))

    if args.dataset in ["mbrset", "all"]:
        # mBRSET has a nested structure in the notebook
        process_mbrset(os.path.join(base_path, "mBRSET/mbrset"))

    if args.dataset in ["ham10000", "all"]:
        process_ham10000(os.path.join(base_path, "HAM10000"))


def run_configured_downloads(config_path: str):
    config = load_config(config_path)
    data_dir = config.get("paths", {}).get("data_dir", "datasets")
    parser = build_parser()

    for item in enabled_items(config.get("download_datasets", [])):
        item_args = item.get("args", [])
        args = parser.parse_args([*item_args, "--data_dir", data_dir])
        print(f"\n--- Downloading {item.get('name', args.dataset)} ---")
        run_download(args)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        run_configured_downloads(args.config)
    else:
        run_download(args)


if __name__ == "__main__":
    main()
