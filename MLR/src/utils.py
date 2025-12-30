"""
utils.py
---------
This file contains helper functions used across the entire Toyota Corolla MLR project.

Functions included:
1. ensure_directory(path)
    → Creates folder if it doesn't exist.

2. save_plot(directory, filename)
    → Saves Matplotlib plots into a folder and closes the figure.

3. calculate_vif(df)
    → Calculates VIF (Variance Inflation Factor) to detect multicollinearity.

4. load_cleaned_data(path)
    → Loads cleaned dataset safely (optional convenience function).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =====================================================================================
# Function: ensure_directory
# =====================================================================================
def ensure_directory(path):
    """
    Ensures that the given directory exists.
    If it does NOT exist, create it.

    Example:
        ensure_directory("output/plots")
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Created directory: {path}")


# =====================================================================================
# Function: save_plot
# =====================================================================================
def save_plot(directory, filename):
    """
    Saves the current Matplotlib plot into the specified directory.

    Parameters:
    directory (str): Folder where the plot should be saved.
    filename (str): Output image filename.

    NOTE:
    plt.close() is used to prevent memory issues and overlapping plots.
    """

    ensure_directory(directory)  # Ensure folder exists
    full_path = os.path.join(directory, filename)

    plt.savefig(full_path)
    plt.close()

    print(f"[INFO] Plot saved: {full_path}")


# =====================================================================================
# Function: calculate_vif
# =====================================================================================
def calculate_vif(df):
    """
    Computes VIF (Variance Inflation Factor) for each numerical feature.

    VIF Purpose:
        - Detects multicollinearity between predictors.
        - High VIF ( > 10 ) means the feature is highly correlated with others.

    Parameters:
    df (DataFrame): Only numeric columns should be passed here.

    Returns:
    vif_df (DataFrame): Columns ['feature', 'VIF']
    """

    print("[INFO] Calculating VIF values...")

    vif_data = []

    # Loop through each column and compute VIF
    for i in range(df.shape[1]):
        vif_value = variance_inflation_factor(df.values, i)
        vif_data.append({
            "feature": df.columns[i],
            "VIF": vif_value
        })

    vif_df = pd.DataFrame(vif_data)
    return vif_df


# =====================================================================================
# Function: load_cleaned_data (optional helper)
# =====================================================================================
def load_cleaned_data(path="data/processed/cleaned_data.csv"):
    """
    Loads cleaned dataset from disk. This is not required but helpful.

    Parameters:
    path (str): File path to cleaned CSV.

    Returns:
    DataFrame or None
    """

    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None

    print(f"[INFO] Loading cleaned dataset from: {path}")
    return pd.read_csv(path)


# =====================================================================================
# Direct File Test
# =====================================================================================
if __name__ == "__main__":
    print("[TEST] utils.py testing...")

    # Test: create folder
    ensure_directory("test_output")

    # Test: fake plot save
    plt.plot([1, 2, 3], [1, 4, 9])
    save_plot("test_output", "test_plot.png")

    # Test: VIF (with dummy data)
    df_test = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],
        "C": [5, 3, 6, 2, 1]
    })
    print(calculate_vif(df_test))
