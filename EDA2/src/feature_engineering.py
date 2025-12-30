import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------
# Function: create_new_features
# Purpose : Generate additional useful features that help a model
#           understand patterns better.
# -------------------------------------------------------------
def create_new_features(df):
    """
    Creates new engineered features to improve model performance.

    New Features:
    1. age_group       -> groups age into categories (young/middle/senior)
    2. capital_net     -> capital_gain minus capital_loss
    """

    # -------- Feature 1: Age Group --------
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 50, 100],  
        labels=["Young", "Middle-Aged", "Senior"]
    )

    # -------- Feature 2: Net Capital --------
    # FIXED: Correct column names based on your dataset
    df["capital_net"] = df["capital_gain"] - df["capital_loss"]

    return df


# -------------------------------------------------------------
# Function: apply_log_transformation
# Purpose : Apply log transformation to reduce skewness in data.
# -------------------------------------------------------------
def apply_log_transformation(df, column_list):
    """
    Applies log(1 + x) transformation to reduce the effect of outliers.
    """

    for col in column_list:
        if col in df.columns:
            df[col] = np.log1p(df[col])  # log(1 + x)

    return df


# -------------------------------------------------------------
# Function: engineer_features
# Purpose : Runs both the new feature creation and log transformation.
#           Saves the engineered dataset to data/processed folder.
# -------------------------------------------------------------
def engineer_features(df, save_path="data/processed/engineered_data.csv"):
    """
    Performs full feature engineering pipeline:
    - Creates new features
    - Applies log transformation to skewed numerical columns
    """

    print("[INFO] Creating new features...")
    df = create_new_features(df)

    print("[INFO] Applying log transformation...")

    # FIXED: Updated to match real dataset column names
    skewed_columns = ["capital_gain", "capital_loss", "capital_net"]

    df = apply_log_transformation(df, skewed_columns)

    # Ensure folder exists before saving
    os.makedirs("data/processed", exist_ok=True)

    # Save engineered dataset
    df.to_csv(save_path, index=False)
    print(f"[INFO] Engineered dataset saved to: {save_path}")

    return df


