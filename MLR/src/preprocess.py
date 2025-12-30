"""
preprocess.py
--------------
This script handles all preprocessing steps required before training the
Multiple Linear Regression model for the Toyota Corolla dataset.

What this script does:
1. Loads raw dataset
2. Cleans missing values
3. Converts categorical data (Fuel_Type) into numeric using One-Hot Encoding
4. Converts "Doors" column into numeric (some datasets have string values)
5. Removes duplicates
6. Saves the cleaned dataset into data/processed/
"""

import os
import pandas as pd
from utils import ensure_directory


# =====================================================================================
# Function: preprocess_data
# =====================================================================================
def preprocess_data(df, save_path="data/processed/cleaned_data.csv"):
    """
    Preprocess the input dataset and save the cleaned version.

    Parameters:
        df (DataFrame): Raw dataset loaded from CSV.
        save_path (str): Output path where cleaned file will be stored.

    Returns:
        cleaned_df (DataFrame)
    """

    print("\n[INFO] Starting Data Preprocessing...")

    # ------------------------------------------------------------------------------
    # 1. Remove Duplicate Rows
    # ------------------------------------------------------------------------------
    print("[INFO] Removing duplicate rows if any...")
    before = df.shape[0]
    df = df.drop_duplicates().copy()
    after = df.shape[0]
    print(f"[INFO] Removed {before - after} duplicate rows.")

    # ------------------------------------------------------------------------------
    # 2. Handle Missing Values
    # ------------------------------------------------------------------------------
    print("[INFO] Checking missing values...")
    print(df.isnull().sum())

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    print("[INFO] Filling numeric missing values with median...")
    for col in numeric_cols:
        df.loc[:, col] = df[col].fillna(df[col].median())

    print("[INFO] Filling categorical missing values with mode...")
    for col in categorical_cols:
        df.loc[:, col] = df[col].fillna(df[col].mode()[0])

    # ------------------------------------------------------------------------------
    # 3. Clean Doors column → Extract numeric part SAFELY
    # ------------------------------------------------------------------------------
    if "Doors" in df.columns:
        print("[INFO] Cleaning 'Doors' column...")

        # Extract digits (handles values like "5doors", "5drs", "nan", etc.)
        df["Doors"] = df["Doors"].astype(str).str.extract(r"(\d+)")

        # Convert extracted values into numbers, invalid extractions become NaN
        df["Doors"] = pd.to_numeric(df["Doors"], errors="coerce")

        # Replace NaN with median number of doors
        median_doors = df["Doors"].median()
        df["Doors"] = df["Doors"].fillna(median_doors)

        # Convert to integer
        df["Doors"] = df["Doors"].astype(int)

    # ------------------------------------------------------------------------------
    # 4. One-Hot Encode Fuel_Type (correct column)
    # ------------------------------------------------------------------------------
    """
    The dataset contains a column 'Fuel_Type'.
    Example values: Petrol, Diesel, CNG.
    """
    if "Fuel_Type" in df.columns:
        print("[INFO] Applying One-Hot Encoding on Fuel_Type...")
        df = pd.get_dummies(df, columns=["Fuel_Type"], drop_first=True)

    # ------------------------------------------------------------------------------
    # 5. Ensure Output Directory Exists
    # ------------------------------------------------------------------------------
    ensure_directory("data/processed")

    # ------------------------------------------------------------------------------
    # 6. Save Cleaned Dataset
    # ------------------------------------------------------------------------------
    print(f"[INFO] Saving cleaned dataset to: {save_path}")
    df.to_csv(save_path, index=False)

    print("[INFO] Preprocessing Completed Successfully!")

    return df


# =====================================================================================
# Standalone Execution for Testing (Optional)
# =====================================================================================
if __name__ == "__main__":
    print("[TEST] Running preprocess.py directly...")

    raw_path = "data/raw/ToyotaCorolla - MLR.csv"

    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        preprocess_data(df_raw)
    else:
        print("[ERROR] Raw dataset not found in data/raw/. Please place the CSV file first.")


