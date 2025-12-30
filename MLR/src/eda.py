"""
eda.py
------
Performs Exploratory Data Analysis (EDA) for the Toyota Corolla MLR project.

This script does:
1. Loads dataset
2. Generates summary statistics
3. Checks missing values
4. Creates histograms for numerical features
5. Creates boxplots to detect outliers
6. Generates a correlation heatmap
7. Saves all outputs into the 'output/' directory
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import helpers from utils.py
# ensure_directory → creates folder if missing
# save_plot → saves and closes matplotlib figure
from utils import ensure_directory, save_plot


# =====================================================================================
# Function: run_eda
# =====================================================================================
def run_eda(df, output_dir="output"):
    """
    Runs full Exploratory Data Analysis (EDA) on the dataset.

    Parameters:
    df (DataFrame) : The loaded dataset.
    output_dir (str) : Folder where EDA outputs will be stored.

    Output:
    - A text report summarizing dataset information
    - Histogram plots
    - Boxplots
    - Correlation heatmap
    """

    print("\n[INFO] Starting Exploratory Data Analysis...")

    # Create folders if they do not exist
    # output_dir → general output folder
    # plots_dir → subfolder specifically for visualizations
    plots_dir = os.path.join(output_dir, "plots")
    ensure_directory(output_dir)
    ensure_directory(plots_dir)

    # ------------------------------------------------------------------------------
    # 1. Basic Dataset Information
    # ------------------------------------------------------------------------------
    print("[INFO] Generating dataset summary...")

    # Path where EDA report text file will be stored
    eda_report_path = os.path.join(output_dir, "eda_report.txt")

    # Open file in write mode to save summary
    with open(eda_report_path, "w") as report:

        # Write title
        report.write("=== EDA REPORT: TOYOTA COROLLA MLR ===\n\n")

        # Shape: number of rows & columns
        report.write(f"Dataset Shape: {df.shape}\n\n")

        # Data types of each column (int, float, object)
        report.write("Column Info:\n")
        report.write(str(df.dtypes))
        report.write("\n\n")

        # Missing values for each column
        report.write("Missing Values:\n")
        report.write(str(df.isnull().sum()))
        report.write("\n\n")

        # Summary statistics (mean, std, min, max, quartiles)
        report.write("Summary Statistics:\n")
        report.write(str(df.describe()))
        report.write("\n\n")

    print(f"[INFO] EDA report saved at: {eda_report_path}")

    # ------------------------------------------------------------------------------
    # 2. Histograms for Numerical Columns
    # ------------------------------------------------------------------------------
    print("[INFO] Creating histograms...")

    # Select only columns that contain numbers
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Loop through each numeric column & generate histogram
    for col in numeric_cols:
        plt.figure(figsize=(7, 4))

        # KDE (Kernel Density) adds smooth probability curve
        sns.histplot(df[col], kde=True)

        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        # Save image file inside output/plots folder
        save_plot(plots_dir, f"histogram_{col}.png")

    # ------------------------------------------------------------------------------
    # 3. Boxplots for Outlier Detection
    # ------------------------------------------------------------------------------
    print("[INFO] Creating boxplots...")

    # Boxplots help identify outliers (values far from normal distribution)
    for col in numeric_cols:
        plt.figure(figsize=(7, 4))
        sns.boxplot(x=df[col])

        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)

        # Save image file
        save_plot(plots_dir, f"boxplot_{col}.png")

    # ------------------------------------------------------------------------------
    # 4. Correlation Heatmap
    # ------------------------------------------------------------------------------
    print("[INFO] Creating correlation heatmap...")

    plt.figure(figsize=(12, 8))

    # Compute correlation matrix
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    # Heatmap shows how strongly variables are related
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

    plt.title("Correlation Heatmap")

    # Save heatmap
    save_plot(plots_dir, "correlation_heatmap.png")

    print("\n[INFO] EDA Completed Successfully!")
    print(f"[INFO] All plots saved in: {plots_dir}")


# =====================================================================================
# Run EDA when executing this file directly
# =====================================================================================
if __name__ == "__main__":
    print("[TEST] Running EDA module directly...")

    # Default expected file location
    raw_path = "data/raw/ToyotaCorolla - MLR.csv"

    # Test execution: load file and run EDA
    if os.path.exists(raw_path):
        df_test = pd.read_csv(raw_path)
        run_eda(df_test)
    else:
        print("[ERROR] Raw dataset not found. Please place the CSV inside data/raw/")

