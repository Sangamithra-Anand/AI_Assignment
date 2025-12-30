"""
load_data.py
------------------------
This file is responsible for:
1. Loading the raw dataset from the data/raw folder.
2. Checking if the file exists.
3. Returning the loaded DataFrame to the pipeline.

Everything is explained step-by-step using comments.
"""

import os
import pandas as pd


# ----------------------------------------------------------------------
# Function: load_raw_dataset
# Purpose : Loads the blogs_categories.csv file from data/raw/
# ----------------------------------------------------------------------
def load_raw_dataset(path="data/raw/blogs_categories.csv"):
    """
    Loads the raw dataset.

    Parameters:
        path (str): Path to the CSV file.

    Returns:
        DataFrame: Loaded dataset if file exists.
        None: If file is missing.
    """

    # Check if the dataset exists before loading
    # This prevents program crashes due to missing files.
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found at: {path}")
        print("[HINT] Make sure blogs_categories.csv is placed inside data/raw/")
        return None

    # Load the dataset using pandas
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)

    # Show basic info about dataset
    print("[INFO] Dataset loaded successfully.")
    print(f"[INFO] Dataset shape: {df.shape}")      # (rows, columns)
    print(f"[INFO] Columns: {list(df.columns)}")    # Column names

    return df



# ----------------------------------------------------------------------
# Self-test block
# Purpose: Allows running this file alone for quick testing.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running load_data.py directly...")
    df_test = load_raw_dataset()

    if df_test is not None:
        print("[TEST] First 5 rows of the dataset:")
        print(df_test.head())
