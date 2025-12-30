"""
visualize.py
---------------------------------------------------------
This file contains helper functions for visualization.
It centralizes all plotting logic so that other modules
(PCA, clustering, comparison) can reuse these functions.

WHAT THIS FILE DOES:
---------------------
1. Plot 2D PCA scatter plot (PC1 vs PC2)
2. Plot side-by-side clustering comparison (Original vs PCA)
3. Plot a generic heatmap (if needed later)

WHY HAVE A SEPARATE VISUALIZATION FILE?
---------------------------------------
- Keeps the project modular
- Avoids repeating plotting code
- All visuals follow a consistent styling
- Makes debugging easier
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Output directory for general visualizations
VISUAL_DIR = "outputs/visuals"
os.makedirs(VISUAL_DIR, exist_ok=True)



# ========================================================================
# FUNCTION: plot_pca_2d(df, labels=None, title="PCA 2D Plot")
# PURPOSE :
#   Visualizes PCA-transformed data using only PC1 and PC2.
# ========================================================================
def plot_pca_2d(df, labels=None, title="PCA 2D Plot"):
    """
    Creates a 2D scatter plot using PC1 and PC2.

    Args:
        df (DataFrame): Should contain columns 'PC1' and 'PC2'
        labels (array-like): Optional cluster labels for coloring
        title (str): Title for the plot

    WHY THIS PLOT?
    --------------
    PC1 and PC2 usually capture the most variance.
    This plot helps visualize:
        ✔ cluster separation
        ✔ high-level structure of PCA data
    """

    if "PC1" not in df.columns or "PC2" not in df.columns:
        raise ValueError("[ERROR] DataFrame must contain PC1 and PC2 columns.")

    plt.figure(figsize=(8, 6))
    plt.scatter(df["PC1"], df["PC2"], c=labels, cmap="viridis")

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    save_path = f"{VISUAL_DIR}/pca_2d_plot.png"
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] PCA 2D plot saved at: {save_path}")



# ========================================================================
# FUNCTION: plot_clusters_comparison(original_df, pca_df, labels_orig, labels_pca)
# PURPOSE :
#   Displays side-by-side cluster plots (Original vs PCA).
# ========================================================================
def plot_clusters_comparison(original_df, pca_df, labels_orig, labels_pca):
    """
    Creates a side-by-side visualization of clustering performance.

    Args:
        original_df (DataFrame): Scaled original dataset
        pca_df (DataFrame): PCA-transformed dataset
        labels_orig (array): Labels from clustering_original.py
        labels_pca (array): Labels from clustering_pca.py

    WHY THIS PLOT?
    --------------
    Helps visually compare:
        ✔ How clusters look BEFORE PCA
        ✔ How clusters look AFTER PCA
    """

    plt.figure(figsize=(14, 6))

    # ---- Plot 1: Original data clustering ----
    plt.subplot(1, 2, 1)
    plt.scatter(original_df.iloc[:, 0], original_df.iloc[:, 1],
                c=labels_orig, cmap="viridis")
    plt.title("Clustering on Original Data")
    plt.xlabel(original_df.columns[0])
    plt.ylabel(original_df.columns[1])

    # ---- Plot 2: PCA clustering ----
    plt.subplot(1, 2, 2)
    plt.scatter(pca_df["PC1"], pca_df["PC2"],
                c=labels_pca, cmap="viridis")
    plt.title("Clustering on PCA Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    save_path = f"{VISUAL_DIR}/cluster_comparison.png"
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Cluster comparison plot saved at: {save_path}")



# ========================================================================
# FUNCTION: plot_heatmap(df, title="Heatmap")
# PURPOSE :
#   Creates a reusable heatmap function.
# ========================================================================
def plot_heatmap(df, title="Heatmap"):
    """
    Creates and saves a heatmap for any dataframe.

    Args:
        df (DataFrame): Should contain numeric values for visualization
        title (str): Title for the heatmap

    WHY THIS FUNCTION?
    ------------------
    Reusable for:
        ✔ Correlation matrices
        ✔ PCA component loadings (future enhancement)
    """

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)

    safe_title = title.replace(" ", "_").lower()
    save_path = f"{VISUAL_DIR}/{safe_title}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Heatmap saved at: {save_path}")



# ========================================================================
# TESTING BLOCK  — runs only when the file is executed directly
# ========================================================================
if __name__ == "__main__":
    print("[TEST] Running visualize.py test mode...")

    try:
        # Create a small fake DataFrame for testing PCA plot
        test_df = pd.DataFrame({
            "PC1": [1, 2, 3, 4],
            "PC2": [4, 3, 2, 1]
        })
        test_labels = [0, 1, 0, 1]

        plot_pca_2d(test_df, labels=test_labels, title="Test PCA Plot")

        print("[TEST] visualize.py executed successfully.")

    except Exception as e:
        print("[TEST ERROR]", e)
