"""
eda.py
---------------------------------------------------------
This file performs Exploratory Data Analysis (EDA) on the dataset.

WHAT THIS FILE DOES:
---------------------
1. Creates summary statistics of the dataset
2. Generates histograms for each numeric feature
3. Generates boxplots for each numeric feature
4. Creates a correlation heatmap
5. Saves all plots inside: outputs/eda_plots/

WHY WE NEED EDA:
----------------
EDA helps you understand:
- The distribution of each feature
- Presence of outliers
- Relationships between variables
- Whether scaling or transformations are required

This step is essential before PCA or clustering.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Create output folder if it doesn't exist
EDA_OUTPUT_DIR = "outputs/eda_plots"
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)



# ========================================================================
# FUNCTION: perform_eda(df)
# PURPOSE :
#   Takes a pandas DataFrame and performs ALL EDA steps.
# ========================================================================
def perform_eda(df):
    """
    Performs full EDA on the given dataset.

    Args:
        df (pandas.DataFrame): The dataset loaded from load_data.py

    This function:
    - Prints summary statistics
    - Saves histograms
    - Saves boxplots
    - Saves correlation heatmap
    """

    print("\n[INFO] Starting EDA...")

    # -------------------------------------------------------------
    # Step 1: Print basic summary statistics
    # describe() gives mean, std, min, max, quartiles for each column.
    # -------------------------------------------------------------
    print("\n[INFO] Summary Statistics:")
    print(df.describe())

    # -------------------------------------------------------------
    # Step 2: Generate histograms for each numeric feature
    # Helps visualize how values are distributed.
    # -------------------------------------------------------------
    print("[INFO] Generating histograms...")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:

        plt.figure(figsize=(6, 4))
        plt.hist(df[col], bins=20, edgecolor='black')
        plt.title(f"Histogram - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        # Save inside outputs/eda_plots/
        plot_path = f"{EDA_OUTPUT_DIR}/histogram_{col}.png"
        plt.savefig(plot_path)
        plt.close()

    print("[INFO] Histogram plots saved.")


    # -------------------------------------------------------------
    # Step 3: Generate boxplots for each numeric feature
    # Boxplots help detect outliers.
    # -------------------------------------------------------------
    print("[INFO] Generating boxplots...")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:

        plt.figure(figsize=(6, 4))
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot - {col}")

        # Save inside outputs/eda_plots/
        plot_path = f"{EDA_OUTPUT_DIR}/boxplot_{col}.png"
        plt.savefig(plot_path)
        plt.close()

    print("[INFO] Boxplot plots saved.")


    # -------------------------------------------------------------
    # Step 4: Correlation Heatmap
    # Helps understand relationships between variables.
    # PCA benefits from knowing which features are strongly correlated.
    # -------------------------------------------------------------
    print("[INFO] Generating correlation heatmap...")

    plt.figure(figsize=(12, 10))
    corr = df.corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")

    heatmap_path = f"{EDA_OUTPUT_DIR}/correlation_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()

    print("[INFO] Correlation heatmap saved.")


    print("\n[INFO] EDA completed successfully!\n")



# ========================================================================
# TESTING BLOCK
# Runs only when file is executed directly
# Helps validate this script before integrating with main.py
# ========================================================================
if __name__ == "__main__":
    from load_data import load_dataset
    
    print("[TEST] Running eda.py directly...")

    try:
        df = load_dataset()          # Load dataset first
        perform_eda(df)              # Run EDA
        print("[TEST] eda.py executed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)



