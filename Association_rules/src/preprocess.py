"""
preprocess.py
--------------------
This file handles ALL preprocessing steps required before running the Apriori
association rule mining algorithm.

What preprocessing means in this project:
1. Clean the dataset (missing values, duplicates, invalid rows).
2. Remove cancelled orders and negative quantities.
3. Keep only necessary columns.
4. Save a cleaned version of the dataset into data/processed/.

This file is executed by main.py, but can also be tested alone.
"""

import os
import pandas as pd


def preprocess_data(df):
    """
    Cleans the raw dataset and prepares it for Association Rule Mining.

    Args:
        df (DataFrame): Raw Online Retail dataset loaded from load_data.py

    Returns:
        DataFrame: Cleaned dataset ready for transformation.
    """

    print("[INFO] Starting preprocessing...")

    # -----------------------------------------------------------
    # STEP 1: Remove rows where essential data is missing
    # -----------------------------------------------------------
    # CustomerID is critical for grouping transactions.
    # Missing IDs cannot be used for basket analysis.
    print("[INFO] Removing rows with missing CustomerID...")
    df = df.dropna(subset=["CustomerID"])

    # Drop rows with missing description also
    print("[INFO] Removing rows with missing product Description...")
    df = df.dropna(subset=["Description"])

    # -----------------------------------------------------------
    # STEP 2: Remove negative or zero quantities
    # -----------------------------------------------------------
    # Negative quantity = product was returned or canceled.
    # Zero quantity = invalid transaction.
    print("[INFO] Removing negative or zero quantities...")
    df = df[df["Quantity"] > 0]

    # -----------------------------------------------------------
    # STEP 3: Remove cancellations using InvoiceNo
    # -----------------------------------------------------------
    # Cancelled transactions contain InvoiceNo starting with 'C'
    print("[INFO] Removing cancelled transactions...")
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # -----------------------------------------------------------
    # STEP 4: Remove duplicate rows
    # -----------------------------------------------------------
    print("[INFO] Removing duplicate rows...")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[INFO] Removed {before - after} duplicate rows.")

    # -----------------------------------------------------------
    # STEP 5: Convert descriptions to clean format
    # -----------------------------------------------------------
    # Remove extra spaces and make text uniform
    print("[INFO] Standardizing product descriptions...")
    df["Description"] = df["Description"].str.strip().str.lower()

    # -----------------------------------------------------------
    # STEP 6: Save cleaned dataset
    # -----------------------------------------------------------
    output_path = "data/processed/cleaned_data.csv"
    df.to_csv(output_path, index=False)

    print(f"[INFO] Preprocessing completed.")
    print(f"[INFO] Cleaned dataset saved to: {output_path}")
    print(f"[INFO] Final dataset shape: {df.shape}")

    return df


# ---------------------------------------------------------
# TEST MODE: Allows testing this file independently
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running preprocess.py directly...")

    # Importing load function without circular import
    from load_data import load_raw_data

    # Load raw dataset
    df_raw = load_raw_data()

    # Preprocess it
    df_clean = preprocess_data(df_raw)

    print("[TEST] First 5 rows of cleaned dataset:")
    print(df_clean.head())


