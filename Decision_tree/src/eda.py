"""
eda.py
-------
This file performs EXPLORATORY DATA ANALYSIS (EDA).

What this script does:
✔ Summary statistics (describe)
✔ Missing value analysis
✔ Safe Correlation heatmap (won't crash if matrix is empty)
✔ Distribution plots for numerical features
✔ Saves all visual outputs into 'reports/' folder

All steps are explained inside the code.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df):
    """
    Perform basic EDA on the cleaned dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataset from preprocess.py

    Returns:
    --------
    None (saves plots to reports/)
    """

    print("\n[INFO] Starting EDA...")

    if df is None:
        print("[ERROR] EDA failed because dataframe is None.")
        return

    # --------------------------------------------------------
    # Ensure reports folder exists
    # --------------------------------------------------------
    reports_path = "reports"
    os.makedirs(reports_path, exist_ok=True)

    # --------------------------------------------------------
    # 1. Summary statistics
    # --------------------------------------------------------
    print("[INFO] Generating summary statistics...")

    describe_path = os.path.join(reports_path, "summary_statistics.csv")
    df.describe().to_csv(describe_path)
    print(f"[INFO] Summary statistics saved: {describe_path}")

    # --------------------------------------------------------
    # 2. Missing value analysis
    # --------------------------------------------------------
    print("[INFO] Checking missing values...")

    missing_values = df.isnull().sum()
    missing_path = os.path.join(reports_path, "missing_values.csv")
    missing_values.to_csv(missing_path)

    print(f"[INFO] Missing value report saved: {missing_path}")

    # --------------------------------------------------------
    # 3. SAFE Correlation heatmap (Fixed crash issue)
    # --------------------------------------------------------
    print("[INFO] Creating correlation heatmap...")

    corr = df.corr(numeric_only=True)

    # Check if correlation matrix is valid
    if corr.shape[0] < 2:
        print("[WARNING] Not enough numeric columns to create heatmap. Skipping...")
    else:
        plt.figure(figsize=(10, 7))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")

        heatmap_path = os.path.join(reports_path, "correlation_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Correlation heatmap saved: {heatmap_path}")

    # --------------------------------------------------------
    # 4. Distribution plots for numerical features
    # --------------------------------------------------------
    print("[INFO] Creating distribution plots...")

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_columns) == 0:
        print("[WARNING] No numeric columns found. Skipping distribution plots...")
    else:
        for col in numeric_columns:
            plt.figure(figsize=(7, 5))
            sns.histplot(df[col], kde=True, color="royalblue")
            plt.title(f"Distribution of {col}")

            dist_path = os.path.join(reports_path, f"dist_{col}.png")
            plt.savefig(dist_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Distribution plot saved for: {col}")

    print("[INFO] EDA completed successfully.\n")


# ============================================================
# TEST: Allow running EDA without full pipeline
# ============================================================
if __name__ == "__main__":
    print("[TEST] Running eda.py directly...")

    from load_data import load_raw_dataset
    from preprocess import preprocess_data

    raw_df = load_raw_dataset()

    if raw_df is not None:
        clean_df = preprocess_data(raw_df)
        run_eda(clean_df)
    else:
        print("[TEST ERROR] Could not run EDA because raw dataset failed to load.")
