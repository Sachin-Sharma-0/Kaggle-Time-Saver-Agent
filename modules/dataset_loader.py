"""
Dataset Loading Module - Converted from Blocks 3, 4, and 10.5
Handles downloading and loading datasets
"""

import pandas as pd
import os
import zipfile
import subprocess
from modules.kaggle_utils import execute_kaggle_command


def download_competition_dataset(competition_name: str, data_dir: str = "./data"):
    """
    Download and extract a Kaggle competition dataset.
    Converted from Block 3.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, f"{competition_name}.zip")
    
    print(f"\nStep 1: Downloading {competition_name} dataset...")
    download_output = execute_kaggle_command([
        "competitions", "download",
        "-c", competition_name,
        "-p", data_dir
    ])
    print(download_output)
    
    print("\nStep 2: Unzipping files...")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        print("✓ Files unzipped successfully.")
        os.remove(zip_path)  # Clean up zip file
    else:
        print(f"❌ Zip file not found: {zip_path}")
        return False
    
    print("\nStep 3: Listing extracted files...")
    files = os.listdir(data_dir)
    for f in files:
        print(f"- {f}")
    
    print("\n✓ Dataset download completed.")
    return True


def load_dataset_from_path(data_dir: str = "./data", train_file: str = "train.csv", 
                           test_file: str = "test.csv", gender_file: str = None):
    """
    Load dataset files from directory.
    Converted from Block 4.
    """
    train_path = os.path.join(data_dir, train_file)
    test_path = os.path.join(data_dir, test_file)
    
    print("\nStep 1: Loading CSV files...")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print("✓ Loaded train and test files.")
    
    gender_df = None
    if gender_file:
        gender_path = os.path.join(data_dir, gender_file)
        if os.path.exists(gender_path):
            gender_df = pd.read_csv(gender_path)
            print("✓ Loaded gender_submission file.")
    
    print("\nStep 2: Dataset shapes:")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    if gender_df is not None:
        print(f"Gender submission shape: {gender_df.shape}")
    
    print("\nStep 3: Train head():")
    print(train_df.head())
    
    print("\nTest head():")
    print(test_df.head())
    
    print("\nStep 4: Missing values in train:")
    print(train_df.isnull().sum())
    
    print("\nStep 5: Basic statistics for numeric columns:")
    print(train_df.describe())
    
    return train_df, test_df, gender_df


def download_kaggle_dataset_simple(kaggle_dataset_id: str, data_dir: str = "./data"):
    """
    Simple function to download a Kaggle dataset.
    Converted from Block 10.5.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    zip_filename = kaggle_dataset_id.split('/')[-1].replace('-', '_') + '.zip'
    zip_filepath = os.path.join(data_dir, zip_filename)
    
    print(f"Downloading {kaggle_dataset_id} dataset...")
    
    def _execute_kaggle_command_download(command_list):
        try:
            result = subprocess.run(
                ["kaggle"] + command_list,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}"
    
    download_output = _execute_kaggle_command_download([
        "datasets", "download",
        "-d", kaggle_dataset_id,
        "-p", data_dir
    ])
    print(download_output)
    
    print("Unzipping files...")
    if os.path.exists(zip_filepath):
        with zipfile.ZipFile(zip_filepath, "r") as z:
            z.extractall(data_dir)
        print("✓ Files unzipped successfully.")
        os.remove(zip_filepath)  # Clean up the zip file
        
        # Verify extraction
        csv_file = os.path.join(data_dir, f"{kaggle_dataset_id.split('/')[-1]}.csv")
        if os.path.exists(csv_file):
            print(f"✓ {os.path.basename(csv_file)} confirmed in data directory.")
        else:
            # Check for any CSV files
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if csv_files:
                print(f"✓ Found CSV files: {csv_files}")
            else:
                print("❌ No CSV files found after unzipping.")
    else:
        print(f"❌ Zip file not found: {zip_filepath}")

