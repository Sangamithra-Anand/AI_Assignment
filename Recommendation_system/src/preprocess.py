"""
preprocess.py
--------------
This file handles all BASIC DATA CLEANING tasks such as:

1. Removing duplicates
2. Handling missing values (NA)
3. Fixing incorrect data types
4. Saving the cleaned dataset into: data/processed/cleaned_anime.csv

IMPORTANT:
- This file does NOT handle feature extraction (TF-IDF, scaling, etc.)
  Feature extraction happens inside feature_engineering.py
"""

import os
import pandas as pd
from utils.helpers import log_message


# -------------------------------------------------------------
# Helper function: ensure processed folder exists
# -------------------------------------------------------------
def ensure_processed_folder(path="data/processed/"):
    """
    Ensures that the folder where cleaned data will be saved actually exists.
    If not, it creates the folder.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        log_message(f"Created folder: {path}", "INFO")


# -------------------------------------------------------------
# Main function: preprocess_dataset
# -------------------------------------------------------------
def preprocess_dataset(df):
    """
    Cleans the raw anime dataset.

    Cleaning Steps:
    1. Remove duplicates
    2. Drop rows with missing essential values (like title or genre)
    3. Convert numeric columns safely
    4. Fill missing numeric values with median
    5. Ensure correct datatypes
    """

    log_message("Starting preprocessing...", "INFO")

    # ---------------------------------------------------------
    # 1. Remove duplicate rows
    # ---------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    log_message(f"Removed {before - after} duplicate rows.", "INFO")

    # ---------------------------------------------------------
    # 2. Drop rows missing titles or genres
    # ---------------------------------------------------------
    df = df.dropna(subset=["name", "genre"], how="any")
    log_message("Dropped rows without title/genre.", "INFO")

    # ---------------------------------------------------------
    # 3. Convert numeric columns safely
    # ---------------------------------------------------------
    numeric_cols = ["rating", "members", "episodes"]

    for col in numeric_cols:
        if col in df.columns:
            # Convert strings, '?', 'Unknown', etc. → NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")
            log_message(f"Converted {col} to numeric (invalid values → NaN).", "INFO")

            # Fill missing values with median
            df[col] = df[col].fillna(df[col].median())
            log_message(f"Filled missing values in '{col}' using median.", "INFO")

    log_message("Preprocessing completed successfully.", "INFO")
    return df


# -------------------------------------------------------------
# Save cleaned dataset to disk
# -------------------------------------------------------------
def save_cleaned_dataset(df, path="data/processed/cleaned_anime.csv"):
    """
    Saves the cleaned dataset into the processed folder.
    """

    ensure_processed_folder(os.path.dirname(path))

    df.to_csv(path, index=False)
    log_message(f"Cleaned dataset saved to: {path}", "INFO")


# -------------------------------------------------------------
# Test block — runs ONLY when executed directly
# -------------------------------------------------------------
if __name__ == "__main__":
    log_message("Running preprocess.py TEST MODE...", "INFO")

    from src.load_data import load_raw_dataset

    raw_df = load_raw_dataset()

    if raw_df is not None:
        cleaned_df = preprocess_dataset(raw_df)
        save_cleaned_dataset(cleaned_df)
        log_message("Preprocess test completed.", "INFO")
    else:
        log_message("Unable to load raw dataset.", "ERROR")

