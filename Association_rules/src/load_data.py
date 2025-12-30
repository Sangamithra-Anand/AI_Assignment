"""
load_data.py
------------------
This module handles loading the Online Retail dataset.

What this file does:
1. Looks for the dataset in the data/raw/ folder.
2. Checks if the file exists (prevents crashes).
3. Loads the Excel file into a pandas DataFrame.
4. Prints helpful information for debugging.
5. Returns the DataFrame so other modules can use it.
"""

import os          # Used to check file paths
import pandas as pd  # Used for loading and handling the dataset


def load_raw_data():
    """
    Loads the Online Retail dataset.

    Returns:
        pandas.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist in the correct folder.
        ValueError: If the file cannot be read properly.
    """

    # -------------------------------
    # STEP 1: Set file path location
    # -------------------------------
    # The dataset must be stored manually by the user in:
    # data/raw/Online_retail.xlsx
    file_path = "data/raw/Online_retail.xlsx"

    print("[INFO] Loading dataset...")

    # -----------------------------------------------
    # STEP 2: Check if the dataset file actually exists
    # -----------------------------------------------
    # This prevents the program from crashing later.
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"[ERROR] Dataset not found at {file_path}. "
            "Please place Online_retail.xlsx inside data/raw/ before running the pipeline."
        )

    # -------------------------------------------
    # STEP 3: Try reading the Excel file
    # -------------------------------------------
    # engine="openpyxl" ensures modern .xlsx files load properly.
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        # If something goes wrong while reading the file,
        # we raise a clear error message.
        raise ValueError(f"[ERROR] Failed to read Excel file: {e}")

    # -----------------------------------------------
    # STEP 4: Print useful dataset information
    # -----------------------------------------------
    # These logs help verify that the dataset is loaded correctly.
    print("[INFO] Dataset loaded successfully.")
    print(f"[INFO] Shape (rows, columns): {df.shape}")
    print(f"[INFO] Column names: {list(df.columns)}")

    # -----------------------------------------------
    # STEP 5: Return the loaded DataFrame
    # -----------------------------------------------
    return df


# -----------------------------------------------------
# TEST MODE: This block only runs when executing the file
# Example: python src/load_data.py
# -----------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running load_data.py directly for testing...")
    df_test = load_raw_data()
    print(df_test.head())    # Show first 5 rows to confirm loading works


