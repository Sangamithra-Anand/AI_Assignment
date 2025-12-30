"""
clustering_original.py
---------------------------------------------------------
This file performs clustering (K-Means) on the ORIGINAL
scaled dataset (before applying PCA).

WHAT THIS FILE DOES:
---------------------
1. Runs K-Means clustering on the scaled data
2. Computes:
      ✔ Silhouette Score
      ✔ Davies–Bouldin Index
3. Visualizes clusters using first two features
4. Saves:
      ✔ Cluster plot
      ✔ JSON report of metrics

WHY WE DO THIS:
---------------
To compare clustering performance BEFORE and AFTER PCA.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


# Output folders for plots + reports
CLUSTER_PLOTS_DIR = "outputs/clustering_plots"
REPORTS_DIR = "outputs/reports"

os.makedirs(CLUSTER_PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)



# ========================================================================
# FUNCTION: run_kmeans_original(df, n_clusters=3)
# PURPOSE :
#   Runs K-Means clustering on the ORIGINAL scaled data.
# ========================================================================
def run_kmeans_original(df, n_clusters=3):
    """
    Applies K-Means clustering on the scaled dataset.

    Args:
        df (pandas.DataFrame): Scaled numeric dataset.
        n_clusters (int): Number of clusters to create.

    Returns:
        dict: Clustering scores (silhouette, DB index)

    STEPS:
        ✔ Fit K-Means
        ✔ Compute cluster labels
        ✔ Evaluate clustering
        ✔ Save cluster plot
        ✔ Save metrics JSON
    """

    print("\n[INFO] Running K-Means clustering on ORIGINAL dataset...")

    # -------------------------------------------------------------
    # Step 1: Initialize K-Means model
    #
    # random_state=42 → ensures consistent results every run.
    # -------------------------------------------------------------
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # -------------------------------------------------------------
    # Step 2: Fit K-Means on the dataset
    #
    # K-Means tries to find "n_clusters" number of groups by minimizing
    # distances between points and cluster centers.
    # -------------------------------------------------------------
    labels = kmeans.fit_predict(df)
    print("[INFO] K-Means model trained successfully.")


    # -------------------------------------------------------------
    # Step 3: Compute evaluation metrics
    #
    # Silhouette Score:
    #   +1 → Well separated clusters
    #    0 → Overlapping clusters
    #   -1 → Wrong clustering
    #
    # Davies–Bouldin Index:
    #   Lower value = better clustering
    # -------------------------------------------------------------
    silhouette = silhouette_score(df, labels)
    db_index = davies_bouldin_score(df, labels)

    print(f"[INFO] Silhouette Score: {silhouette:.4f}")
    print(f"[INFO] Davies–Bouldin Index: {db_index:.4f}")


    # -------------------------------------------------------------
    # Step 4: Visualize clusters
    #
    # Since original data has many dimensions, we plot the first two
    # features for visualization.
    #
    # NOTE: This plot does NOT represent real clusters perfectly
    #       because real separation happens in higher dimensions.
    # -------------------------------------------------------------
    print("[INFO] Generating cluster visualization...")

    plt.figure(figsize=(7, 5))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap="viridis")
    plt.title("K-Means Clustering (Original Scaled Data)")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])

    plot_path = f"{CLUSTER_PLOTS_DIR}/original_kmeans.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Cluster plot saved at: {plot_path}")


    # -------------------------------------------------------------
    # Step 5: Save clustering scores to JSON
    # -------------------------------------------------------------
    scores = {
        "silhouette_score": float(silhouette),
        "davies_bouldin_index": float(db_index),
        "n_clusters": n_clusters
    }

    json_path = f"{REPORTS_DIR}/clustering_original_scores.json"
    with open(json_path, "w") as f:
        json.dump(scores, f, indent=4)

    print(f"[INFO] Clustering metrics saved at: {json_path}")


    print("\n[INFO] Clustering on original dataset completed!\n")
    return scores



# ========================================================================
# TESTING BLOCK — allows running this file independently
# ========================================================================
if __name__ == "__main__":
    from load_data import load_dataset
    from preprocess import preprocess_data

    print("[TEST] Running clustering_original.py directly...")

    try:
        df_raw = load_dataset()           # Load raw data
        df_scaled = preprocess_data(df_raw)  # Scale it
        run_kmeans_original(df_scaled)       # Run clustering
        print("[TEST] clustering_original.py executed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)

