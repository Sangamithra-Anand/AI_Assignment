"""
load_data.py
----------------------
This module loads the Zoo dataset from the data folder.

Everything is explained inside the code using comments.
"""

import os
import pandas as pd


# =========================================================
# FUNCTION: create_output_folders()
# Purpose :
#     - Automatically create all output folders used by the project.
#     - This avoids the need to manually create "output/models", etc.
#     - exist_ok=True prevents errors if the folder already exists.
# =========================================================
def create_output_folders():
    folders = [
        "output",
        "output/plots",
        "output/models",
        "output/reports"
    ]

    # Create each folder one by one
    for folder in folders:
        os.makedirs(folder, exist_ok=True)  # Auto-create if missing


# =========================================================
# FUNCTION: load_zoo_data()
# Purpose :
#     - Load Zoo.csv from the data folder.
#     - Validate that the file exists.
#     - Print dataset details.
#     - Return the DataFrame for use in other modules.
# =========================================================
def load_zoo_data(filepath="data/Zoo.csv"):
    """
    Parameters:
        filepath : str
            Path to the Zoo dataset.

    Returns:
        df : pandas DataFrame
            The loaded dataset.
    """

    print("[INFO] Preparing output directories...")
    create_output_folders()   # Ensures output folders exist before anything else

    print(f"[INFO] Loading dataset from: {filepath}")

    # ----------------------------
    # Step 1: Check if file exists
    # ----------------------------
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        print("[HINT] Place Zoo.csv inside the 'data/' folder.")
        return None  # Return None instead of crashing

    # ----------------------------
    # Step 2: Try loading the CSV
    # ----------------------------
    try:
        df = pd.read_csv(filepath)      # Read dataset into a DataFrame
        print("[INFO] Dataset loaded successfully.")
        print(f"[INFO] Dataset shape: {df.shape}")    # (rows, columns)
        print(f"[INFO] Columns: {list(df.columns)}")  # List all column names
        return df

    except Exception as e:
        # If pandas fails to read the file (bad format, encoding, etc.)
        print(f"[ERROR] Failed to read CSV file: {e}")
        return None


# =========================================================
# TESTING BLOCK
# Runs only when you execute:
#     python src/load_data.py
#
# Purpose:
#     - Helps confirm that file loading works before continuing.
#     - Does NOT run when imported in main.py.
# =========================================================
if __name__ == "__main__":
    print("[TEST] Running load_data.py directly...")

    df_test = load_zoo_data()  # Try loading dataset

    if df_test is not None:
        print("[TEST] File loaded successfully during direct test.")
    else:
        print("[TEST ERROR] Could not load dataset.")

