"""
preprocess.py
--------------
This file handles PREPROCESSING of the raw dataset.

What this script does:
✔ Receives the loaded raw dataset
✔ Cleans missing values
✔ Removes duplicates
✔ Handles incorrect column names
✔ Saves cleaned dataset into data/processed/
✔ Returns the cleaned DataFrame

Every operation is explained inside the code.
"""

import os
import pandas as pd


def preprocess_data(df, save=True):
    """
    Preprocess the dataset step-by-step.

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset loaded from load_data.py

    save : bool
        If True → saves cleaned CSV inside data/processed/

    Returns:
    --------
    cleaned_df : pandas.DataFrame
    """

    print("\n[INFO] Starting preprocessing...")

    # -------------------------------
    # 1. Check if dataset is None
    # -------------------------------
    if df is None:
        print("[ERROR] Cannot preprocess because dataset is None.")
        return None

    # -------------------------------
    # 2. Remove duplicate rows
    # -------------------------------
    print("[INFO] Removing duplicate rows...")
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"[INFO] Duplicates removed: {before - after}")

    # -------------------------------
    # 3. Standardize column names
    # -------------------------------
    print("[INFO] Standardizing column names...")
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    print(f"[INFO] Updated columns: {list(df.columns)}")

    # -------------------------------
    # 4. Handle missing values
    # -------------------------------
    print("[INFO] Checking missing values...")
    missing_count = df.isnull().sum().sum()

    if missing_count > 0:
        print(f"[WARNING] Missing values found: {missing_count}")
        print("[INFO] Filling missing values with column mean (numerical only)...")

        # Fills only numerical columns with mean
        df = df.fillna(df.mean(numeric_only=True))

        # Fill categorical missing values with mode
        df = df.fillna(df.mode().iloc[0])
    else:
        print("[INFO] No missing values found.")

    # -------------------------------
    # 5. Ensure processed data folder exists
    # -------------------------------
    processed_path = os.path.join("data", "processed")
    os.makedirs(processed_path, exist_ok=True)

    # -------------------------------
    # 6. Save cleaned dataset
    # -------------------------------
    if save:
        output_file = os.path.join(processed_path, "cleaned_data.csv")
        df.to_csv(output_file, index=False)
        print(f"[INFO] Cleaned dataset saved to: {output_file}")

    print("[INFO] Preprocessing completed successfully.\n")

    return df


# ============================================================
# TEST: Allow this file to run independently (optional)
# ============================================================
if __name__ == "__main__":
    print("[TEST] Running preprocess.py directly...")

    from load_data import load_raw_dataset

    raw_df = load_raw_dataset()

    if raw_df is not None:
        cleaned = preprocess_data(raw_df)
        print("[TEST] Cleaned dataset preview:")
        print(cleaned.head())
    else:
        print("[TEST ERROR] Could not load dataset for preprocessing.")
