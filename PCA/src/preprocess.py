"""
preprocess.py
---------------------------------------------------------
This file handles ALL preprocessing steps required before
running PCA or clustering.

WHAT THIS FILE DOES:
---------------------
1. Removes non-numeric columns (if present)
2. Handles missing values
3. Scales/standardizes numeric features
4. Saves the processed dataset to data/processed/
5. Returns the cleaned & scaled DataFrame

WHY WE NEED PREPROCESSING:
--------------------------
- PCA requires scaled data (mean = 0, std = 1)
- K-means clustering also works better on scaled data
- Cleaning ensures PCA and clustering do not fail
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Output folder for processed data
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)



# ========================================================================
# FUNCTION: preprocess_data(df)
# PURPOSE :
#   Takes raw dataset → cleans → scales → returns processed dataset.
# ========================================================================
def preprocess_data(df):
    """
    Cleans and scales the dataset.

    Args:
        df (pandas.DataFrame): Raw dataset loaded from load_data.py

    Returns:
        pandas.DataFrame: Scaled numeric dataset

    STEPS:
        ✔ Remove non-numeric columns
        ✔ Handle missing values
        ✔ Apply StandardScaler
        ✔ Save processed CSV
    """

    print("\n[INFO] Starting preprocessing...")

    # -------------------------------------------------------------
    # Step 1: Select only numeric columns
    # PCA and clustering CANNOT run on strings or labels.
    # -------------------------------------------------------------
    print("[INFO] Selecting numeric columns only...")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    print(f"[INFO] Columns selected: {list(numeric_df.columns)}")



    # -------------------------------------------------------------
    # Step 2: Handle missing values
    # Strategy: Fill missing values with column mean
    #
    # WHY?
    # - PCA cannot run with NaN values
    # - Mean imputation is simple and effective for numeric features
    # -------------------------------------------------------------
    print("[INFO] Checking for missing values...")

    if numeric_df.isnull().sum().sum() > 0:
        print("[WARNING] Missing values detected. Filling with column means.")
        numeric_df = numeric_df.fillna(numeric_df.mean())
    else:
        print("[INFO] No missing values found.")



    # -------------------------------------------------------------
    # Step 3: Standardize data using StandardScaler
    #
    # WHY SCALING IS REQUIRED:
    # - PCA gives more importance to large-scale features
    # - StandardScaler → mean = 0, std = 1
    # - Very important for K-means distance calculations
    # -------------------------------------------------------------
    print("[INFO] Scaling numeric features...")

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(numeric_df)

    # Convert scaled numpy array back to DataFrame
    scaled_df = pd.DataFrame(scaled_array, columns=numeric_df.columns)

    print("[INFO] Scaling completed.")




    # -------------------------------------------------------------
    # Step 4: Save processed data for later use
    # -------------------------------------------------------------
    save_path = f"{PROCESSED_DATA_DIR}/scaled_wine.csv"
    scaled_df.to_csv(save_path, index=False)

    print(f"[INFO] Processed dataset saved at: {save_path}")



    print("\n[INFO] Preprocessing completed successfully!\n")
    return scaled_df



# ========================================================================
# TESTING BLOCK (runs only when this file is executed directly)
# Helps validate preprocessing before integrating with main.py
# ========================================================================
if __name__ == "__main__":
    from load_data import load_dataset

    print("[TEST] Running preprocess.py directly...")

    try:
        raw_df = load_dataset()            # Load dataset
        clean_df = preprocess_data(raw_df) # Preprocess it
        print("[TEST] preprocess.py executed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)

