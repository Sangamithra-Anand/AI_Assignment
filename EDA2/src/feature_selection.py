import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# -------------------------------------------------------------
# Function: detect_outliers
# Purpose : Identify unusual datapoints using Isolation Forest.
# -------------------------------------------------------------
def detect_outliers(df, contamination=0.02):
    """
    Detects outliers using Isolation Forest.

    Returns:
        df with an 'outlier' column (1 = normal, -1 = outlier)
    """
    print("[INFO] Running Isolation Forest...")

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    df["outlier"] = model.fit_predict(numeric_df)

    return df


# -------------------------------------------------------------
# Function: remove_outliers
# Purpose : Remove rows labeled as outliers (-1).
# -------------------------------------------------------------
def remove_outliers(df):
    """
    Removes rows marked as outliers.

    Returns:
        cleaned_df without outliers
    """
    print("[INFO] Removing outliers...")

    cleaned_df = df[df["outlier"] == 1].copy()
    cleaned_df.drop(columns=["outlier"], inplace=True)

    print(f"[INFO] Outlier-free dataset shape: {cleaned_df.shape}")

    return cleaned_df


# -------------------------------------------------------------
# Function: compute_mutual_information
# Purpose : Calculate Mutual Information for all features
#           relative to the target column.
# -------------------------------------------------------------
def compute_mutual_information(df, target_column):
    """
    Computes Mutual Information for all features relative to the target.
    Converts ALL categorical columns to numeric before MI calculation.
    """

    print("[INFO] Computing Mutual Information scores...")

    # Separate target from features
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify categorical & numeric columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Convert ALL categorical features into numeric codes
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes

    # Convert target if it’s categorical
    if y.dtype == 'object' or str(y.dtype) == 'category':
        y = y.astype('category').cat.codes

    # Choose MI function based on target type
    if y.dtype in ['int64', 'float64']:
        mi_scores = mutual_info_regression(X, y)
    else:
        mi_scores = mutual_info_classif(X, y)

    # Build MI table
    mi_df = pd.DataFrame({
        "feature": X.columns,
        "mutual_information": mi_scores
    }).sort_values(by="mutual_information", ascending=False)

    return mi_df


# -------------------------------------------------------------
# Function: feature_selection_pipeline
# Purpose : Full feature selection:
#           - detect & remove outliers
#           - compute MI matrix
# -------------------------------------------------------------
def feature_selection_pipeline(df,
                               target_column="income",
                               outlier_path="output/outliers_removed.csv",
                               mi_path="output/mutual_information.csv"):
    """
    Full feature selection pipeline using Isolation Forest + MI scores
    """
    os.makedirs("output", exist_ok=True)

    # Step 1: Detect outliers
    df_out = detect_outliers(df)

    # Step 2: Remove outliers
    df_cleaned = remove_outliers(df_out)
    df_cleaned.to_csv(outlier_path, index=False)
    print(f"[INFO] Outlier-free dataset saved to: {outlier_path}")

    # Step 3: Mutual Information analysis
    mi_df = compute_mutual_information(df_cleaned, target_column)
    mi_df.to_csv(mi_path, index=False)
    print(f"[INFO] Mutual Information scores saved to: {mi_path}")

    return df_cleaned, mi_df
