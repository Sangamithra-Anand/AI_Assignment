"""
load_data.py
------------
This file is responsible for:
1. Loading the raw anime dataset from data/raw/
2. Performing quick validation checks
3. Returning a clean DataFrame to the pipeline

The code is written with detailed explanations so that
any beginner can understand exactly what each part does.

NOTE:
- This file does NOT clean the dataset. It ONLY loads
  and validates the raw CSV.
- Full cleaning happens in preprocess.py
"""

import os
import pandas as pd


# -------------------------------------------------------------
# Function: load_raw_dataset
# Purpose : Load anime.csv from the raw data folder.
# -------------------------------------------------------------
def load_raw_dataset(path="data/raw/anime.csv"):
    """
    Loads the raw anime dataset from the specified path.

    Parameters:
        path (str): file path for anime.csv

    Returns:
        pandas.DataFrame: raw dataset if file exists
        None: if file is missing

    Explanation:
    - We first check whether the dataset exists using os.path.exists()
    - If it does not exist, we display an error message.
    - If it exists, we load it into a pandas DataFrame.
    """

    # Check whether the dataset exists
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found at: {path}")
        print("Please place anime.csv inside data/raw/ folder.")
        return None

    # Load the dataset into a DataFrame
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)

    # Display basic info for developer understanding
    print("[INFO] Dataset loaded successfully.")
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    return df


# -------------------------------------------------------------
# Test block
# Purpose : When this file is run directly (python load_data.py),
#           it will test loading the dataset.
# -------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running load_data.py directly...")

    df = load_raw_dataset()

    if df is not None:
        print("[TEST] First 5 rows of the dataset:")
        print(df.head())
    else:
        print("[TEST ERROR] Unable to load dataset. Check file path.")
