"""
data_loader.py
--------------
This file handles:
- Loading the raw CSV file (Alphabets_data.csv)
- Showing basic information about the dataset
- Returning summary details to be used in other scripts

Everything is explained inside the code itself.
"""

import os
from typing import Dict, Any

import pandas as pd

# Import required settings and paths from config.py
from config import (
    RAW_DATA_PATH,      # Full path to raw Alphabets_data.csv
    TARGET_COLUMN,      # Name of the target label column (example: "label")
    create_directories   # Function to auto-create all folders
)


def load_raw_data() -> pd.DataFrame:
    """
    Loads the raw dataset from the CSV file located in data/raw/.

    Why this function is needed:
    - We want a reusable loading function for preprocessing, training, etc.
    - We want to show clear errors if the dataset is missing.

    Returns:
        A pandas DataFrame containing the raw dataset.
    """

    # Check if the dataset exists before trying to load it.
    # This prevents crashes and gives clean error messages.
    if not os.path.exists(RAW_DATA_PATH):
        # RAW_DATA_PATH is built in config.py
        raise FileNotFoundError(
            f"[ERROR] Raw data file not found at: {RAW_DATA_PATH}\n"
            "Place 'Alphabets_data.csv' inside the 'data/raw/' folder."
        )

    # Let the user know where the file is being loaded from.
    print(f"[INFO] Loading raw dataset from: {RAW_DATA_PATH}")

    # Load the CSV using pandas.
    # pd.read_csv converts CSV → DataFrame (table-like structure)
    df = pd.read_csv(RAW_DATA_PATH)

    # Confirm loading + show shape
    print("[INFO] Dataset loaded successfully.")
    print(f"[INFO] Dataset shape: {df.shape}")  # Example: (1000, 20)

    return df
    # Note: df is now a DataFrame that other functions can process.


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Creates a summary of the important dataset details.

    Why this function:
    - To quickly understand basic dataset structure before model building.
    - Useful for debugging or checking if preprocessing is needed.

    Returns:
        A dictionary containing rows, columns, missing values, target info.
    """

    # Get number of rows and columns (shape returns (rows, cols))
    n_rows, n_cols = df.shape

    # Convert DataFrame column names to a simple list
    columns = df.columns.tolist()

    # Count missing values per column (very important for preprocessing)
    missing_per_column = df.isna().sum().to_dict()

    # Information about the target label column
    target_info = {}

    # Check if the target column actually exists
    if TARGET_COLUMN in df.columns:
        # Number of unique classes (example: 26 alphabets)
        n_classes = df[TARGET_COLUMN].nunique()

        # Show top 10 most common alphabet labels
        value_counts = df[TARGET_COLUMN].value_counts().head(10).to_dict()

        target_info = {
            "target_column": TARGET_COLUMN,
            "n_classes": n_classes,
            "value_counts_top10": value_counts,
        }
    else:
        # If dataset does not contain expected target column, warn
        target_info = {
            "target_column": TARGET_COLUMN,
            "warning": f"Target column '{TARGET_COLUMN}' not found!"
        }

    # Build and return the overview dictionary
    overview = {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "columns": columns,
        "missing_per_column": missing_per_column,
        "target_info": target_info,
    }

    return overview


def print_dataset_overview(overview: Dict[str, Any]) -> None:
    """
    Prints the dataset overview in a nice, readable format.

    Why separate function?
    - Clean separation of logic (computing overview vs. printing it)
    - Reusable for debugging and checking data integrity
    """

    print("\n==================== DATASET OVERVIEW ====================\n")

    print(f"Number of rows   : {overview['n_rows']}")
    print(f"Number of columns: {overview['n_columns']}\n")

    print("Column names:")
    for col in overview["columns"]:
        print(f"  - {col}")
    print()

    print("Missing values per column:")
    for col, n_miss in overview["missing_per_column"].items():
        print(f"  {col}: {n_miss}")
    print()

    print("Target column info:")
    for key, value in overview["target_info"].items():
        print(f"  {key}: {value}")
    print("\n==========================================================\n")


# ---------------------------------------------------------------------
# This block only runs if you execute:
#     python data_loader.py
# It will NOT run when imported from another file.
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Ensure all folders exist (created automatically)
    create_directories()

    try:
        # 2. Load dataset from CSV
        df_raw = load_raw_data()

        # 3. Build overview dictionary
        overview_dict = get_dataset_overview(df_raw)

        # 4. Print cleaned overview to the terminal
        print_dataset_overview(overview_dict)

    except FileNotFoundError as e:
        # If dataset missing, show clear message
        print(e)


