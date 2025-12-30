import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df_numeric, save_path="data/processed/cleaned_data.csv"):
    """
    Preprocess the numeric dataset for clustering.

    Steps performed:
    -----------------------------------------------------------
    1. Remove duplicate rows
    2. Handle missing values (replace with column median)
    3. Remove outliers using percentile clipping (1% - 99%)
    4. Scale features using StandardScaler
    5. Save cleaned + scaled data to CSV

    Returns:
        df_clean      → cleaned (but unscaled) DataFrame
        df_scaled     → scaled data as a DataFrame
    """

    print("\n[INFO] Starting preprocessing...")

    # -----------------------------------------------------------
    # 1. REMOVE DUPLICATES
    # -----------------------------------------------------------
    initial_rows = df_numeric.shape[0]
    df_clean = df_numeric.drop_duplicates()
    removed = initial_rows - df_clean.shape[0]
    print(f"[INFO] Removed duplicate rows: {removed}")

    # -----------------------------------------------------------
    # 2. HANDLE MISSING VALUES
    #
    # Using median because it is robust against outliers.
    # -----------------------------------------------------------
    df_clean = df_clean.fillna(df_clean.median())
    print("[INFO] Missing values handled using column medians.")

    # -----------------------------------------------------------
    # 3. OUTLIER REMOVAL (Clipping)
    #
    # Why clipping?
    # - Keeps extreme values under control without deleting rows.
    # - Prevents KMeans / Hierarchical from being distorted.
    #
    # Method:
    # - Values below 1st percentile -> set to 1st percentile
    # - Values above 99th percentile -> set to 99th percentile
    # -----------------------------------------------------------
    lower = df_clean.quantile(0.01)
    upper = df_clean.quantile(0.99)
    df_clean = df_clean.clip(lower, upper, axis=1)
    print("[INFO] Outliers clipped to 1% - 99% range.")

    # -----------------------------------------------------------
    # 4. FEATURE SCALING
    #
    # Why scaling?
    # - Clustering distance-based algorithms require scaling
    # - Prevents large-scale features from dominating
    #
    # StandardScaler converts values to:
    #   mean = 0, std = 1
    # -----------------------------------------------------------
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_clean)

    df_scaled = pd.DataFrame(
        scaled_array,
        columns=df_clean.columns
    )

    print("[INFO] Scaling completed using StandardScaler.")

    # -----------------------------------------------------------
    # 5. SAVE CLEANED DATASET (SCALED)
    #
    # Save into: data/processed/cleaned_data.csv
    # If folder does not exist → error message tells user to create.
    # -----------------------------------------------------------
    processed_dir = os.path.dirname(save_path)

    if not os.path.exists(processed_dir):
        print(f"[WARNING] Directory {processed_dir} does not exist. Creating it...")
        os.makedirs(processed_dir, exist_ok=True)

    df_scaled.to_csv(save_path, index=False)
    print(f"[INFO] Cleaned dataset saved to: {save_path}")

    return df_clean, df_scaled


# -----------------------------------------------------------
# TEST RUN (executed only if you run this file directly)
# -----------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running preprocess.py directly...")

    try:
        # Quick test: try loading numeric df using load_data.py
        from load_data import load_dataset
        df_raw, df_numeric = load_dataset()

        preprocess_data(df_numeric)

        print("[TEST] preprocess.py executed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)


