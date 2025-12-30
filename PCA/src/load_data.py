"""
load_data.py
---------------------------------------------------------
This file ONLY handles loading the dataset.

WHY WE NEED THIS FILE:
- To keep the project clean and modular.
- Other scripts (EDA, PCA, clustering) should not load the data themselves.
- This script loads the dataset, checks basic issues, and returns it.

This makes the entire project easier to debug and maintain.
"""

import os          # Used to check if the dataset file exists
import pandas as pd  # Used to load CSV files and work with data frames


# ========================================================================
# FUNCTION: load_dataset()
# PURPOSE :
#   1. Load the dataset from data/raw/
#   2. Validate file existence
#   3. Print useful dataset information for the user
#   4. Return the loaded DataFrame to be used by other modules
#
# WHY A FUNCTION?
#   - Easy to reuse
#   - Easy to test independently
#   - Helps main.py call this function anytime
# ========================================================================
def load_dataset(path="data/raw/wine.csv"):
    """
    Loads the dataset and performs basic validation.

    Args:
        path (str): File path to the dataset.
                    Default = "data/raw/wine.csv"

    Returns:
        pandas.DataFrame: The loaded dataset.

    This function does:
      ✔ Check if path exists
      ✔ Load CSV file
      ✔ Print shape, column names, sample rows
      ✔ Check for missing values
    """

    # -------------------------------------------------------------
    # Step 1: Tell the user where the dataset is being loaded from
    # -------------------------------------------------------------
    print("[INFO] Loading dataset from:", path)

    # -------------------------------------------------------------
    # Step 2: Check if the file actually exists.
    # This prevents the program from crashing later.
    # -------------------------------------------------------------
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found at: {path}")

    # -------------------------------------------------------------
    # Step 3: Load the CSV file into a pandas DataFrame.
    # pd.read_csv converts CSV into a table-like object we can use.
    # -------------------------------------------------------------
    df = pd.read_csv(path)
    print("[INFO] Dataset loaded successfully.")

    # -------------------------------------------------------------
    # Step 4: Print dataset shape
    # Example: (178, 13) → 178 rows, 13 columns
    # This helps confirm the dataset is correct.
    # -------------------------------------------------------------
    print("[INFO] Dataset shape:", df.shape)

    # -------------------------------------------------------------
    # Step 5: Print the column names
    # Helps you understand the structure of the data.
    # -------------------------------------------------------------
    print("[INFO] Columns:", list(df.columns))

    # -------------------------------------------------------------
    # Step 6: Show first 5 rows
    # Helps verify if the data loaded correctly and looks normal.
    # -------------------------------------------------------------
    print("\n[INFO] First 5 rows of the dataset:")
    print(df.head())

    # -------------------------------------------------------------
    # Step 7: Check for missing values
    # Missing values can break PCA, clustering, and scaling.
    # So we check early.
    # -------------------------------------------------------------
    missing_count = df.isnull().sum().sum()

    if missing_count > 0:
        print(f"\n[WARNING] Missing values detected: {missing_count}")
    else:
        print("\n[INFO] No missing values found.")

    # -------------------------------------------------------------
    # Step 8: Return the dataset so other files can use it.
    # -------------------------------------------------------------
    return df


# ========================================================================
# TESTING BLOCK (runs only when this file is executed directly)
#
# WHY THIS BLOCK?
#   - You can run:  python src/load_data.py
#   - This helps you test the loading logic BEFORE integrating with main.py
# ========================================================================
if __name__ == "__main__":
    print("[TEST] Running load_data.py directly...")

    try:
        df = load_dataset()  # Attempt to load the dataset
        print("\n[TEST] load_data.py executed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)


