"""
load_data.py
------------
This file is responsible for LOADING the dataset.

✔ Loads the raw dataset from data/raw/
✔ Validates the dataset
✔ Returns the DataFrame for preprocessing
✔ Prints info messages for debugging
✔ Gracefully handles errors (missing file, wrong format)

Every line of code is explained inside this script.
"""

import os
import pandas as pd


def load_raw_dataset(path="data/raw/heart_disease.csv"):
    """
    Loads the dataset from data/raw/ folder.

    Parameters:
    -----------
    path : str
        Full path to the dataset file (default: data/raw/heart_disease.csv)

    Returns:
    --------
    df : pandas.DataFrame or None
        Returns the dataset as a DataFrame.
        Returns None if file is missing or corrupted.
    """

    # Print the path so user knows exactly what file is being loaded
    print(f"[INFO] Loading dataset from: {path}")

    # ------------------------------------------------------------
    # 1. Check if file exists
    # ------------------------------------------------------------
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        print("[HINT] Make sure your dataset exists inside data/raw/")
        return None

    try:
        # --------------------------------------------------------
        # 2. Read CSV file (we are now using heart_disease.csv)
        # --------------------------------------------------------
        df = pd.read_csv(path)
        print("[INFO] Dataset loaded successfully.")

        # Print basic dataset info
        print(f"[INFO] Dataset shape: {df.shape}")
        print(f"[INFO] Columns: {list(df.columns)}")

        return df

    except Exception as e:
        # --------------------------------------------------------
        # 3. Error handling (in case CSV is corrupted or unreadable)
        # --------------------------------------------------------
        print("[ERROR] Could not read the dataset.")
        print(f"[DETAIL] {e}")
        return None


# ============================================================
# TEST: Allow direct execution for debugging
# ============================================================
if __name__ == "__main__":
    print("[TEST] Running load_data.py directly...")

    df = load_raw_dataset()

    if df is not None:
        print("[TEST] First 5 rows of dataset:")
        print(df.head())
    else:
        print("[TEST ERROR] Dataset could not be loaded.")
