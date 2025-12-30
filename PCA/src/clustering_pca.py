"""
clustering_pca.py
---------------------------------------------------------
This file performs K-Means clustering on the PCA-transformed
dataset and evaluates the results.

WHAT THIS FILE DOES:
---------------------
1. Loads PCA-transformed dataset
2. Runs K-Means clustering
3. Computes:
      ✔ Silhouette Score
      ✔ Davies–Bouldin Index
4. Visualizes clusters using PC1 and PC2
5. Saves:
      ✔ Cluster scatter plot
      ✔ JSON metrics summary

WHY THIS STEP IS IMPORTANT:
---------------------------
We compare clustering performance:
    BEFORE PCA  → clustering_original.py
    AFTER PCA   → clustering_pca.py

This helps understand the effect of dimensionality reduction on clustering.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


# Output folders for PCA clustering plots + reports
CLUSTER_PLOTS_DIR = "outputs/clustering_plots"
REPORTS_DIR = "outputs/reports"
PROCESSED_DIR = "data/processed"

os.makedirs(CLUSTER_PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)



# ========================================================================
# FUNCTION: run_kmeans_pca(df, n_clusters=3)
# PURPOSE :
#   Runs K-Means clustering on PCA-transformed data.
# ========================================================================
def run_kmeans_pca(df, n_clusters=3):
    """
    Applies K-Means clustering on PCA-transformed dataset.

    Args:
        df (pandas.DataFrame): DataFrame containing PCA components (PC1, PC2, ...)
        n_clusters (int): Number of clusters

    Returns:
        dict: Dictionary of evaluation metrics
    """

    print("\n[INFO] Running K-Means clustering on PCA-transformed data...")

    # -------------------------------------------------------------
    # Step 1: Initialize K-Means
    # -------------------------------------------------------------
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # -------------------------------------------------------------
    # Step 2: Fit K-Means on PCA dataset
    #
    # PCA reduces high dimensions → clustering becomes easier,
    # faster, and sometimes more accurate.
    # -------------------------------------------------------------
    labels = kmeans.fit_predict(df)
    print("[INFO] K-Means model trained successfully.")


    # -------------------------------------------------------------
    # Step 3: Compute evaluation metrics
    #
    # Silhouette Score:
    #   +1 = excellent clustering
    #    0 = overlapping clusters
    #   -1 = poor clustering
    #
    # Davies–Bouldin Index:
    #   Lower = better
    # -------------------------------------------------------------
    silhouette = silhouette_score(df, labels)
    db_index = davies_bouldin_score(df, labels)

    print(f"[INFO] Silhouette Score (PCA): {silhouette:.4f}")
    print(f"[INFO] Davies–Bouldin Index (PCA): {db_index:.4f}")


    # -------------------------------------------------------------
    # Step 4: Visualize clusters using first two principal components
    #
    # NOTE:
    #   PCA already reduces dimensions.
    #   PC1 and PC2 together typically capture most information.
    # -------------------------------------------------------------
    print("[INFO] Creating PCA cluster visualization...")

    plt.figure(figsize=(7, 5))
    plt.scatter(df["PC1"], df["PC2"], c=labels, cmap="viridis")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("K-Means Clustering on PCA-Transformed Data")

    plot_path = f"{CLUSTER_PLOTS_DIR}/pca_kmeans.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] PCA cluster plot saved at: {plot_path}")


    # -------------------------------------------------------------
    # Step 5: Save metrics to JSON report
    # -------------------------------------------------------------
    scores = {
        "silhouette_score": float(silhouette),
        "davies_bouldin_index": float(db_index),
        "n_clusters": n_clusters
    }

    json_path = f"{REPORTS_DIR}/clustering_pca_scores.json"
    with open(json_path, "w") as f:
        json.dump(scores, f, indent=4)

    print(f"[INFO] PCA clustering metrics saved at: {json_path}")


    print("\n[INFO] PCA clustering completed.\n")
    return scores



# ========================================================================
# TESTING BLOCK — executes when running this file directly
# ========================================================================
if __name__ == "__main__":
    from load_data import load_dataset
    from preprocess import preprocess_data
    from pca_model import apply_pca

    print("[TEST] Running clustering_pca.py directly...")

    try:
        raw_df = load_dataset()           # Load raw dataset
        scaled_df = preprocess_data(raw_df)  # Scale numeric data
        pca_df = apply_pca(scaled_df)        # PCA transformation
        run_kmeans_pca(pca_df)               # Cluster PCA data

        print("[TEST] clustering_pca.py executed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)

