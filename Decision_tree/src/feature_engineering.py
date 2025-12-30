"""
feature_engineering.py
-----------------------
This file handles FEATURE ENGINEERING for the dataset.

What this script now does:
✔ Label Encoding (for ordinal / simple categories)
✔ One-Hot Encoding (for nominal categories)
✔ Scaling numerical features using StandardScaler or MinMaxScaler
✔ Saves a report of all transformations
✔ Returns the transformed DataFrame

All steps include explanations inside the code.
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


def apply_label_encoding(df, columns):
    """
    Applies Label Encoding to specified columns.
    Converts string categories to numeric labels.
    """
    print("\n[INFO] Applying Label Encoding...")

    le = LabelEncoder()

    for column in columns:
        if column in df.columns:
            df[column] = le.fit_transform(df[column].astype(str))
            print(f"[INFO] Label Encoded column: {column}")
        else:
            print(f"[WARNING] Column not found: {column}")

    return df


def apply_one_hot_encoding(df, columns):
    """
    Applies One-Hot Encoding to convert categorical variables 
    into multiple binary columns.
    """
    print("\n[INFO] Applying One-Hot Encoding...")

    df = pd.get_dummies(df, columns=columns, drop_first=True)
    print(f"[INFO] One-Hot Encoding applied to: {columns}")

    return df


def apply_scaling(df, columns, method="standard"):
    """
    Applies scaling to numerical columns.

    Parameters:
    -----------
    columns : list
        List of numerical columns to scale
    method : str
        "standard" → StandardScaler
        "minmax"   → MinMaxScaler
    """

    print("\n[INFO] Applying Feature Scaling...")

    if method == "standard":
        scaler = StandardScaler()
        print("[INFO] Using StandardScaler (mean=0, SD=1).")
    elif method == "minmax":
        scaler = MinMaxScaler()
        print("[INFO] Using MinMaxScaler (0 to 1).")
    else:
        print("[ERROR] Invalid scaling method. Choose 'standard' or 'minmax'.")
        return df

    # Only scale columns that exist in df
    valid_cols = [col for col in columns if col in df.columns]

    if not valid_cols:
        print("[WARNING] No valid columns found for scaling.")
        return df

    # Fit + transform scaler
    df[valid_cols] = scaler.fit_transform(df[valid_cols])

    print(f"[INFO] Scaled columns: {valid_cols}")
    return df


def feature_engineering(df, label_encode_cols=None, one_hot_cols=None,
                        scale_cols=None, scale_method="standard",
                        save_report=True):
    """
    Applies CLEAN + ENCODE + SCALE transformations on dataset.

    Parameters:
    -----------
    label_encode_cols : list
    one_hot_cols : list
    scale_cols : list
        Numerical columns to scale
    scale_method : str
        "standard" or "minmax"
    """

    print("\n[INFO] Starting Feature Engineering...")

    if df is None:
        print("[ERROR] Dataset is None. Cannot apply feature engineering.")
        return None

    original_cols = list(df.columns)

    # ----------------------------------------------------------
    # 1. Label Encoding
    # ----------------------------------------------------------
    if label_encode_cols:
        df = apply_label_encoding(df, label_encode_cols)

    # ----------------------------------------------------------
    # 2. One-Hot Encoding
    # ----------------------------------------------------------
    if one_hot_cols:
        df = apply_one_hot_encoding(df, one_hot_cols)

    # ----------------------------------------------------------
    # 3. Scaling Numerical Features
    # ----------------------------------------------------------
    if scale_cols:
        df = apply_scaling(df, scale_cols, method=scale_method)

    # ----------------------------------------------------------
    # 4. Save feature engineering report
    # ----------------------------------------------------------
    if save_report:
        reports_path = "reports"
        os.makedirs(reports_path, exist_ok=True)

        report_file = os.path.join(reports_path, "feature_engineering_report.txt")

        with open(report_file, "w") as f:
            f.write("===== FEATURE ENGINEERING REPORT =====\n\n")

            f.write("Original Columns:\n")
            f.write(str(original_cols) + "\n\n")

            f.write("Label Encoded Columns:\n")
            f.write(str(label_encode_cols) + "\n\n")

            f.write("One-Hot Encoded Columns:\n")
            f.write(str(one_hot_cols) + "\n\n")

            f.write("Scaled Columns:\n")
            f.write(f"{scale_cols} using method '{scale_method}'\n\n")

            f.write("Final Columns After All Transformations:\n")
            f.write(str(list(df.columns)) + "\n")

        print(f"[INFO] Feature Engineering report saved: {report_file}")

    print("[INFO] Feature Engineering completed successfully.\n")
    return df


# ============================================================
# TEST MODE
# ============================================================
if __name__ == "__main__":
    print("[TEST] Running feature_engineering.py directly...")

    from load_data import load_raw_dataset
    from preprocess import preprocess_data

    raw_df = load_raw_dataset()

    if raw_df is not None:
        clean_df = preprocess_data(raw_df)

        # CHANGE THESE BASED ON YOUR DATASET
        label_cols = ["sex"]          # example
        onehot_cols = ["cp"]          # example
        scale_columns = ["age", "chol", "trestbps"]  # example numeric columns

        final_df = feature_engineering(
            clean_df,
            label_encode_cols=label_cols,
            one_hot_cols=onehot_cols,
            scale_cols=scale_columns,
            scale_method="standard"
        )

        print("[TEST] Feature Engineering preview:")
        print(final_df.head())
