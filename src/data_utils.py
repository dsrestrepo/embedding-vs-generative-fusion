import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

######### Datasets Preparation #########
def preprocess_df(df, image_columns, images_path, coco_format=False):
    # Function to check if an image can be opened
    def is_valid_image(img_path):
        img_path = os.path.join(images_path, img_path)
        # img_path = str(int(img_path)) + ".jpeg"
        # print(img_path)
        
        try:
            # Image.open(img_path).convert("RGB")
            # print(img_path)
            return True
        except:
            
            print("invalid path!")
            return False

    # Function to correct image paths without extensions
    def correct_image_path(img_path):
        if type(img_path) != str:
            img_path = str(int(img_path))
            if coco_format:
                if len(img_path) < 12:
                    img_path = '0' * (12 - len(img_path)) + img_path + ".jpg"
    
        full_img_path = os.path.join(images_path, img_path)
        img_path, file_name = os.path.split(full_img_path)

        if '.' not in file_name:
            # Try to find the correct extension in the directory
            for file in os.listdir(img_path):
                if file.split('.')[0] == file_name:
                    return os.path.join(images_path, file) 
        return full_img_path
    
    # remove elements with missing image paths
    df = df[df[image_columns].notna()]

    # Correct image paths if necessary
    df[image_columns] = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(correct_image_path)(img_path) for img_path in tqdm(df[image_columns]))

    # Filter out rows with images that cannot be opened
    valid_mask = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(is_valid_image)(img_path) for img_path in tqdm(df[image_columns]))
    df = df[valid_mask]

    # Correct image paths if necessary
    #df[image_columns] = df[image_columns].apply(correct_image_path)

    # Filter out rows with images that cannot be opened
    #df = df[df[image_columns].apply(is_valid_image)]

    return df

######### Merge Datasets #########
# Define a function to check if a value is a list
def is_list(value):
    return isinstance(value, str)


def process_embeddings(df, col_name):
    """
    Process embeddings in a DataFrame column.

    Args:
    - df (pd.DataFrame): The DataFrame containing the embeddings column.
    - col_name (str): The name of the column containing the embeddings.

    Returns:
    pd.DataFrame: The DataFrame with processed embeddings.

    Steps:
    1. Convert the values in the specified column to lists.
    2. Extract values from lists and create new columns for each element.
    3. Remove the original embeddings column.

    Example:
    df_processed = process_embeddings(df, 'embeddings')
    """
    print("S: check str")
    # Apply the function to each element in the 'embeddings' column
    mask = df['embeddings'].apply(is_list)

    # Filter out the rows where the elements are not lists
    df = df[mask]
    # Step 1: Convert the values in the column to lists
    df[col_name] = df[col_name].apply(eval)

    # Step 2-4: Extract values from lists and create new columns
    embeddings_df = pd.DataFrame(df[col_name].to_list(), columns=[f"text_{i+1}" for i in range(df[col_name].str.len().max())])
    df = pd.concat([df, embeddings_df], axis=1)

    # Step 5: Remove the original "embeddings" column
    df = df.drop(columns=[col_name])

    return df

def rename_image_embeddings(df):
    """
    Rename columns in a DataFrame for image embeddings.

    Args:
    - df (pd.DataFrame): The DataFrame containing columns to be renamed.

    Returns:
    pd.DataFrame: The DataFrame with renamed columns.

    Example:
    df_renamed = rename_image_embeddings(df)
    """
    df.columns = [f'image_{int(col)}' if col.isdigit() else col for col in df.columns]

    return df

# Preprocess and merge the dataframes
def preprocess_data(text_data, image_data, text_id="image_id", image_id="ImageName", embeddings_col = 'embeddings'):
    """
    Preprocess and merge text and image dataframes.

    Args:
    - text_data (pd.DataFrame): DataFrame containing text data.
    - image_data (pd.DataFrame): DataFrame containing image data.
    - text_id (str): Column name for text data identifier.
    - image_id (str): Column name for image data identifier.
    - embeddings_col (str): Column name for embeddings data.

    Returns:
    pd.DataFrame: Merged and preprocessed DataFrame.

    Steps:
    1. Process text and image embeddings.
    2. Convert image_id and text_id values to integers.
    3. Merge dataframes using image_id.
    4. Drop unnecessary columns.

    Example:
    merged_df = preprocess_data(text_df, image_df)
    """
    text_data = process_embeddings(text_data, embeddings_col)
    image_data = rename_image_embeddings(image_data)
    
    # # Remove file extension from image_id
    # if text_data[text_id].dtype != int:
    #     text_data[text_id] = text_data[text_id].apply(lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x.split('.')[0])
    # if image_data[image_id].dtype != int:
    #     image_data[image_id] = image_data[image_id].apply(lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x.split('.')[0])
    text_data[text_id] = text_data[text_id].apply(lambda x: str(x).split('.')[0] if isinstance(x, str) else x)

    # text_data[text_id] = text_data[text_id].apply(lambda x: x.split('.')[0])
    image_data[image_id] = image_data[image_id].apply(lambda x: x.split('.')[0])

    # Merge dataframes using image_id
    df = pd.merge(text_data, image_data, left_on=text_id, right_on=image_id)
    
    # df = pd.concat([text_data, image_data], axis=0)

    # Drop unnecessary columns
    df.drop([image_id, text_id], axis=1, inplace=True)

    return df

# Function to split the data into train and test
def split_data(df, val_size=None, random_state=42, stratify_col=None):
    """
    Split a DataFrame into train and test sets based on the 'split' column.
    Can also split train into train/val if val_size is provided.

    Args:
    - df (pd.DataFrame): The DataFrame to be split.
    - val_size (float, optional): Proportion of training set to use as validation set.

    Returns:
    pd.DataFrame: Train set (or Train/Val if val_size provided)
    pd.DataFrame: Test set.
    """
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    if 'val' in df['split'].unique():
        val_df = df[df['split'] == 'val']
        print("Train Shape:", train_df.shape)
        print("Val Shape:", val_df.shape)
        print("Test Shape:", test_df.shape)
        return train_df, val_df, test_df

    
    if val_size:
        from sklearn.model_selection import train_test_split
        if stratify_col:
            train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state, stratify=train_df[stratify_col])
        else:
            train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)
        print("Train Shape:", train_df.shape)
        print("Val Shape:", val_df.shape)
        print("Test Shape:", test_df.shape)
        return train_df, val_df, test_df
    
    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)
    return train_df, test_df

# Function to process text labels and one-hot encode them
from sklearn.preprocessing import MultiLabelBinarizer

class DummyMLB:
    def __init__(self, classes):
        self.classes_ = classes

def process_labels(df, col='answer', mlb=None, train_columns=None):
    """
    Process text labels and perform one-hot encoding using MultiLabelBinarizer.
    If col is a list of strings, it assumes these are already binary columns and returns them.
    """
    if 'DR_3' in df.columns:
        df.DR_3 = df.DR_3.astype(str)
    
    # CASE 1: multilabel columns provided directly
    if isinstance(col, list):
        one_hot_labels = df[col]
        # Create a dummy MLB object to satisfy the return signature
        if mlb is None:
            mlb = DummyMLB(classes=col)
            train_columns = col
            return one_hot_labels, mlb, train_columns
        else:
            return one_hot_labels

    if mlb is None:
        mlb = MultiLabelBinarizer()

        if df[col].dtype == int:
            label = df[col]
            labels = df[col].apply(lambda x: [str(x)])
        else:
            labels = df[col].astype(str).apply(lambda x: set(x.split(', ')))
        
        if df[col].dtype == int and (len(df[col].unique()) == 2):
            train_columns = col
            one_hot_labels = label
        else:
            one_hot_labels = pd.DataFrame(mlb.fit_transform(labels), columns=mlb.classes_)
            # Save the columns from the training set
            train_columns = one_hot_labels.columns
        
        return one_hot_labels, mlb, train_columns

    else:
        if df[col].dtype == int:
            label = df[col]
            labels = df[col].apply(lambda x: [str(x)])
        else:
            labels = df[col].astype(str).apply(lambda x: set(x.split(', ')))
        
        if df[col].dtype == int and (len(df[col].unique()) == 2):
            one_hot_labels = label
        else:
            one_hot_labels = pd.DataFrame(mlb.transform(labels), columns=train_columns)
        
        return one_hot_labels
