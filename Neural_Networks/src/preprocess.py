"""
preprocess.py
-------------
This file is responsible for preparing the dataset for the ANN model.

Main steps:
1. Load the raw dataset (CSV) using data_loader.py
2. Handle missing values
3. Encode categorical features (if any)
4. Scale numeric features (very important for Neural Networks)
5. Save the final processed dataset into data/processed/

All important parts are explained inside the code.
"""

import os

import pandas as pd
from sklearn.preprocessing import StandardScaler  # used for feature scaling

# We reuse settings and helper functions from config.py
from config import (
    INTERIM_DATA_DIR,        # Folder where we can save a "cleaned but not scaled" version
    PROCESSED_DATA_PATH,     # Final processed CSV path
    TARGET_COLUMN,           # Name of the target label column (example: "label")
    create_directories       # Makes sure all folders exist
)

# We reuse the raw data loading function from data_loader.py
from data_loader import load_raw_data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform full preprocessing on the raw dataset.

    Steps done here:
    - Remove rows where the target label is missing
    - Separate features (X) and target (y)
    - Handle missing values in features
    - One-hot encode categorical columns (if any)
    - Scale numeric columns
    - Reattach the target column at the end

    Args:
        df: Raw dataset loaded from the CSV file.

    Returns:
        processed_df: Fully processed DataFrame ready for model training.
    """

    # -----------------------------
    # 1. Drop rows with missing target
    # -----------------------------
    # We cannot train a model if we don't know the correct label (target),
    # so any row with missing target is removed.
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"[ERROR] Target column '{TARGET_COLUMN}' not found in dataset.\n"
            "Please check TARGET_COLUMN in config.py."
        )

    # Drop rows where target is NaN
    df = df.dropna(subset=[TARGET_COLUMN])
    print(f"[INFO] After dropping rows with missing target: {df.shape}")

    # -----------------------------
    # 2. Split into features (X) and target (y)
    # -----------------------------
    # X = all columns except the target
    # y = target column only
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"[INFO] Number of feature columns before preprocessing: {X.shape[1]}")

    # -----------------------------
    # 3. Identify numeric and categorical columns
    # -----------------------------
    # Numeric columns: int, float
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Categorical columns: everything that is not numeric
    # (object, string, category, etc.)
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    print(f"[INFO] Numeric feature columns   : {numeric_cols}")
    print(f"[INFO] Categorical feature columns: {categorical_cols}")

    # -----------------------------
    # 4. Handle missing values
    # -----------------------------
    # For numeric columns: fill missing values with the column median
    # (median is more robust to outliers than mean).
    for col in numeric_cols:
        if X[col].isna().sum() > 0:
            median_value = X[col].median()
            X[col] = X[col].fillna(median_value)
            print(f"[INFO] Filled missing values in numeric column '{col}' with median={median_value}")

    # For categorical columns: fill missing values with the most frequent value (mode).
    for col in categorical_cols:
        if X[col].isna().sum() > 0:
            mode_value = X[col].mode().iloc[0]  # .mode() returns a Series
            X[col] = X[col].fillna(mode_value)
            print(f"[INFO] Filled missing values in categorical column '{col}' with mode='{mode_value}'")

    # -----------------------------
    # 5. Save a cleaned (but not yet scaled) version to interim folder
    # -----------------------------
    # This step is optional, but useful for debugging and checking the cleaning.
    # We build the path using INTERIM_DATA_DIR from config.py.
    interim_path = os.path.join(INTERIM_DATA_DIR, "alphabets_cleaned.csv")
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)  # ensure folder exists
    cleaned_df_for_save = X.copy()
    cleaned_df_for_save[TARGET_COLUMN] = y.values  # reattach target for this version
    cleaned_df_for_save.to_csv(interim_path, index=False)
    print(f"[INFO] Saved cleaned (unscaled) data to: {interim_path}")

    # -----------------------------
    # 6. One-Hot Encode categorical columns (if any)
    # -----------------------------
    # Neural networks work with numeric values only,
    # so we need to convert categorical features into numeric.
    # We use pandas.get_dummies for simple one-hot encoding.
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
        print(f"[INFO] After one-hot encoding, number of feature columns: {X.shape[1]}")
    else:
        print("[INFO] No categorical columns found. Skipping one-hot encoding.")

    # -----------------------------
    # 7. Scale numeric columns
    # -----------------------------
    # Scaling is VERY important for ANNs because:
    # - It helps gradient descent converge faster
    # - It prevents features with large ranges from dominating the loss
    #
    # We use StandardScaler:
    #   z = (x - mean) / std
    #
    # Important: we must identify which columns in X are numeric AFTER
    # one-hot encoding, since new dummy columns are also numeric.
    numeric_cols_after_encoding = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Create a scaler object
    scaler = StandardScaler()

    # Fit the scaler on numeric columns and transform them
    X[numeric_cols_after_encoding] = scaler.fit_transform(X[numeric_cols_after_encoding])
    print(f"[INFO] Scaled {len(numeric_cols_after_encoding)} numeric feature columns.")

    # -----------------------------
    # 8. Reattach the target column to the processed features
    # -----------------------------
    processed_df = X.copy()
    processed_df[TARGET_COLUMN] = y.values

    print(f"[INFO] Final processed dataset shape: {processed_df.shape}")

    return processed_df


def save_processed_data(df_processed: pd.DataFrame) -> None:
    """
    Saves the fully processed dataset to the processed data path.

    The path is defined in config.py as PROCESSED_DATA_PATH.

    Args:
        df_processed: DataFrame returned by preprocess_data().
    """
    # Ensure the folder for processed data exists
    processed_dir = os.path.dirname(PROCESSED_DATA_PATH)
    os.makedirs(processed_dir, exist_ok=True)

    # Save as CSV
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[INFO] Saved processed data to: {PROCESSED_DATA_PATH}")


# ---------------------------------------------------------------------
# If we run this file directly using:
#     python preprocess.py
# then the following block will execute.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """
    This part allows you to run preprocessing as a standalone step.

    Running:
        python preprocess.py

    Will:
    1. Create all required directories (data, models, reports, output, etc.)
    2. Load the raw dataset from data/raw/Alphabets_data.csv
    3. Preprocess the data (cleaning, encoding, scaling)
    4. Save the final processed dataset into data/processed/
    """

    # 1. Ensure folder structure exists
    create_directories()

    # 2. Load the raw dataset
    try:
        raw_df = load_raw_data()
    except FileNotFoundError as e:
        # If the CSV is missing, print error and stop
        print(e)
    else:
        # 3. Preprocess the dataset
        processed_df = preprocess_data(raw_df)

        # 4. Save the processed dataset
        save_processed_data(processed_df)


