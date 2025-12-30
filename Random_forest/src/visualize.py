"""
visualize.py
-------------
This file generates all visualizations needed for the Glass dataset.

Plots included:
1. Histograms (distribution of each feature)
2. Boxplots (detecting outliers)
3. Correlation Heatmap (relationship between features)
4. Pairplot (optional - useful but heavy)

All plots will be saved inside the folder:
outputs/histograms/
outputs/boxplots/
outputs/heatmaps/

The folder structure is AUTO-CREATED if missing.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Helper: Create necessary output folders
# -------------------------------------------------------------------------
def ensure_output_folders():
    """
    Creates folders for saving visualizations if they do not exist.

    Folders created:
    - outputs/
    - outputs/histograms/
    - outputs/boxplots/
    - outputs/heatmaps/

    Reason:
    -------
    Prevents FileNotFoundError and keeps project organized.
    """
    folders = [
        "outputs",
        "outputs/histograms",
        "outputs/boxplots",
        "outputs/heatmaps",
    ]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[AUTO] Created folder: {folder}")


# -------------------------------------------------------------------------
# HISTOGRAMS
# -------------------------------------------------------------------------
def plot_histograms(df):
    """
    Generates histograms for every numerical column.

    Why:
    ----
    Helps understand data distribution:
    - Normal?
    - Skewed?
    - Bimodal?

    Saves plots to: outputs/histograms/
    """
    print("[INFO] Generating histograms...")

    for col in df.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col], bins=20, color="skyblue", edgecolor="black")
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        save_path = f"outputs/histograms/{col}_hist.png"
        plt.savefig(save_path)
        plt.close()

    print("[INFO] Histograms saved successfully.")


# -------------------------------------------------------------------------
# BOXPLOTS
# -------------------------------------------------------------------------
def plot_boxplots(df):
    """
    Generates boxplots for each feature.

    Why:
    ----
    Helps detect:
    - Outliers
    - Spread of data
    - Skewness

    Saves plots to: outputs/boxplots/
    """
    print("[INFO] Generating boxplots...")

    for col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col], color="lightcoral")
        plt.title(f"Boxplot of {col}")

        save_path = f"outputs/boxplots/{col}_boxplot.png"
        plt.savefig(save_path)
        plt.close()

    print("[INFO] Boxplots saved successfully.")


# -------------------------------------------------------------------------
# CORRELATION HEATMAP
# -------------------------------------------------------------------------
def plot_correlation_heatmap(df):
    """
    Generates a correlation heatmap.

    Why:
    ----
    - Shows how features relate to one another
    - Highlights multicollinearity
    - Helps with feature selection

    Saves plot to: outputs/heatmaps/correlation_heatmap.png
    """
    print("[INFO] Generating correlation heatmap...")

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")

    save_path = "outputs/heatmaps/correlation_heatmap.png"
    plt.savefig(save_path)
    plt.close()

    print("[INFO] Correlation heatmap saved successfully.")


# -------------------------------------------------------------------------
# MAIN FUNCTION — Run All Visualizations
# -------------------------------------------------------------------------
def run_visualizations(df):
    """
    Runs all visualization functions.

    Steps:
    ------
    1. Create output folders
    2. Generate histograms
    3. Generate boxplots
    4. Generate heatmap

    Pairplot is optional — can be enabled if needed.
    """
    print("\n[INFO] Starting visualization pipeline...")

    ensure_output_folders()
    plot_histograms(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)

    print("[INFO] All visualizations generated successfully!")


# -------------------------------------------------------------------------
# TEST BLOCK — Run file alone using:
# python src/visualize.py
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running visualize.py directly...")

    try:
        df_test = pd.read_excel("data/raw/glass.xlsx")
        run_visualizations(df_test)
        print("\n[TEST] visualize.py is working correctly ✔️")
    except Exception as e:
        print(f"[TEST ERROR] {e}")
