"""
preprocess.py
----------------------
This module handles preprocessing of the Zoo dataset.

It performs:
    1. Checking for missing values
    2. Handling outliers (optional step)
    3. Feature scaling (VERY important for KNN)

All steps are explained inside the code using detailed comments.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# =========================================================
# FUNCTION: check_missing_values()
# Purpose :
#     - Print missing value counts for each column.
#     - Warn user if missing values must be handled.
# =========================================================
def check_missing_values(df):
    print("\n[INFO] Checking for missing values...")

    missing_count = df.isnull().sum()

    if missing_count.sum() == 0:
        print("[INFO] No missing values found.")
    else:
        print("[WARNING] Missing values detected:")
        print(missing_count)
        print("[HINT] You may need to fill or drop missing values.")


# =========================================================
# FUNCTION: detect_outliers()
# Purpose :
#     - Simple Z-score method to detect outliers.
#     - Outliers are values that deviate more than 3 standard deviations.
#     - OPTIONAL: You can remove or keep them depending on dataset.
# =========================================================
def detect_outliers(df, threshold=3):
    print("\n[INFO] Detecting outliers using Z-score...")

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # Calculate Z-scores for each numeric feature
    z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())

    # Mark outliers (True/False)
    outlier_mask = (z_scores > threshold)

    total_outliers = outlier_mask.sum().sum()

    print(f"[INFO] Total outliers detected: {total_outliers}")

    return outlier_mask


# =========================================================
# FUNCTION: scale_features()
# Purpose :
#     - KNN uses distance calculations.
#     - Non-numeric columns (like "animal name") must NOT be scaled.
#     - StandardScaler normalizes values to mean=0, std=1.
# =========================================================
def scale_features(df, target_column):
    print("\n[INFO] Scaling features using StandardScaler...")

    scaler = StandardScaler()

    # Identify columns that should NOT be scaled
    # (all non-numeric except target)
    non_numeric_cols = list(df.select_dtypes(include=['object']).columns)

    # Remove target from this list
    if target_column in non_numeric_cols:
        non_numeric_cols.remove(target_column)

    # Example: ["animal name"]
    ignored_columns = non_numeric_cols.copy()

    # ----------------------------------------------
    # STEP 1: Separate numeric features for scaling
    # ----------------------------------------------
    X = df.drop(columns=[target_column] + ignored_columns, errors="ignore")
    y = df[target_column]

    # Fit scaler on numeric data
    X_scaled = scaler.fit_transform(X)

    print("[INFO] Feature scaling completed.")

    # ----------------------------------------------
    # STEP 2: Reconstruct DataFrame
    # ----------------------------------------------
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Add ignored columns back (like animal name)
    for col in ignored_columns:
        if col in df.columns:
            scaled_df[col] = df[col]

    # Add target column back
    scaled_df[target_column] = y

    return scaled_df


# =========================================================
# FUNCTION: preprocess_data()
# Purpose :
#     - Complete preprocessing pipeline:
#         * Check missing values
#         * Detect outliers
#         * Scale features
#     - Returns cleaned + scaled DataFrame
# =========================================================
def preprocess_data(df, target_column="type"):
    print("\n[INFO] Starting preprocessing pipeline...")

    # Step 1: Missing value check
    check_missing_values(df)

    # Step 2: Outlier detection (OPTIONAL)
    detect_outliers(df)

    # Step 3: Feature scaling (IMPORTANT for KNN)
    df_scaled = scale_features(df, target_column)

    print("[INFO] Preprocessing completed successfully.")

    return df_scaled


# =========================================================
# TESTING BLOCK
# Runs only when executing:
#     python src/preprocess.py
# =========================================================
if __name__ == "__main__":
    print("[TEST] Testing preprocess.py...")

    try:
        df_sample = pd.read_csv("data/Zoo.csv")
        processed = preprocess_data(df_sample)

        print("[TEST] Preprocessing finished. Sample output:")
        print(processed.head())

    except Exception as e:
        print("[TEST ERROR] Could not run preprocess test:", e)


