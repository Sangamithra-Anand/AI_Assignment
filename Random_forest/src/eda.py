"""
eda.py
------
This file performs Exploratory Data Analysis (EDA) on the Glass dataset.

Tasks covered:
1. Display dataset shape and column names
2. Show data types of each column
3. Display summary statistics (mean, std, min, max...)
4. Check missing values
5. Compute correlation matrix
6. Save the full EDA report into: reports/eda_report.txt

Notes:
- This file does NOT create plots (handled in visualize.py)
- This focuses only on textual EDA analysis.
"""

import os
import pandas as pd


# -------------------------------------------------------------------------
# Helper function: Auto-create reports folder
# -------------------------------------------------------------------------
def ensure_reports_folder(path="reports/"):
    """
    Auto-creates the 'reports' folder if missing.

    Reason:
    -------
    Many users forget to create the folder manually.
    This prevents "FileNotFoundError" when saving the report.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[AUTO] Created folder: {path}")


# -------------------------------------------------------------------------
# Main Function: Run EDA
# -------------------------------------------------------------------------
def run_eda(df):
    """
    Performs Exploratory Data Analysis on the dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
         Raw or preprocessed dataset

    Returns:
    --------
    report_text : str
        The full EDA report (also saved to a file)

    Explanation of Steps:
    ---------------------
    1. Dataset shape
    2. Column names
    3. Data types
    4. Summary statistics
    5. Missing values
    6. Correlation matrix
    """

    print("\n[INFO] Starting EDA...")

    # 1. Dataset Shape -----------------------------------------------------
    shape_info = f"Dataset Shape: {df.shape}\n"

    # 2. Column Names ------------------------------------------------------
    column_info = f"Columns: {list(df.columns)}\n"

    # 3. Data Types --------------------------------------------------------
    dtypes_info = f"Data Types:\n{df.dtypes}\n\n"

    # 4. Summary Statistics ------------------------------------------------
    describe_info = f"Summary Statistics:\n{df.describe()}\n\n"

    # 5. Missing Values ----------------------------------------------------
    missing_info = f"Missing Values:\n{df.isnull().sum()}\n\n"

    # 6. Correlation Matrix ------------------------------------------------
    corr_info = f"Correlation Matrix:\n{df.corr()}\n\n"

    # ---------------------------------------------------------------------
    # Combine everything into a single EDA report
    # ---------------------------------------------------------------------
    report_text = (
        "==================== EDA REPORT ====================\n\n"
        + shape_info
        + column_info
        + dtypes_info
        + describe_info
        + missing_info
        + corr_info
        + "=====================================================\n"
    )

    # Print key parts to console
    print("[INFO] Dataset shape:", df.shape)
    print("[INFO] Columns:", list(df.columns))
    print("[INFO] Missing values:\n", df.isnull().sum())

    # ---------------------------------------------------------------------
    # Save EDA report to reports folder
    # ---------------------------------------------------------------------
    ensure_reports_folder()

    report_path = "reports/eda_report.txt"

    with open(report_path, "w") as file:
        file.write(report_text)

    print(f"[INFO] EDA report saved to: {report_path}")

    return report_text


# -------------------------------------------------------------------------
# TEST BLOCK — Run this file standalone using:
# python src/eda.py
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running eda.py directly...")

    try:
        df_test = pd.read_excel("data/raw/glass.xlsx")
        run_eda(df_test)
        print("\n[TEST] eda.py is working correctly ✔️")
    except Exception as e:
        print(f"[TEST ERROR] {e}")
