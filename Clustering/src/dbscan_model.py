import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


def run_dbscan(
    X_scaled,
    df_original,
    k_distance_plot_path="output/plots/dbscan_kdist.png",
    labels_path="output/labels/dbscan_labels.csv",
    eps=None,
    min_samples=5
):
    """
    Performs DBSCAN clustering.

    Steps:
    ------------------------------------------------------------
    1. Generate K-distance plot to help choose EPS
    2. If EPS is not provided, select automatically
    3. Run DBSCAN with eps + min_samples
    4. Compute silhouette score (ignoring noise)
    5. Save cluster labels

    Args:
        X_scaled           → Scaled numeric data (numpy or DataFrame)
        df_original        → Original DataFrame
        k_distance_plot_path → Path to save k-distance plot
        labels_path        → Path to save DBSCAN labels
        eps                → Neighborhood radius (None = auto pick)
        min_samples        → Minimum points required to form a cluster

    Returns:
        eps_used         → eps used for final model
        labels           → cluster labels
        silhouette_value → silhouette score (ignoring noise)
    """

    print("\n[INFO] Starting DBSCAN clustering...")

    # ------------------------------------------------------------
    # 1. GENERATE K-DISTANCE PLOT
    # ------------------------------------------------------------
    print(f"[INFO] Generating k-distance plot (k={min_samples - 1})...")

    k = min_samples - 1
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)

    # Sort distances to find the "elbow"
    distances = np.sort(distances[:, k - 1])

    plt.figure(figsize=(6, 4))
    plt.plot(distances)
    plt.title(f"k-distance Plot (k={k})")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {k}th nearest neighbor")
    plt.grid(True)

    os.makedirs(os.path.dirname(k_distance_plot_path), exist_ok=True)
    plt.savefig(k_distance_plot_path)
    plt.close()

    print(f"[INFO] K-distance plot saved to: {k_distance_plot_path}")

    # ------------------------------------------------------------
    # 2. AUTO-SELECT EPS IF NOT PROVIDED
    # ------------------------------------------------------------
    if eps is None:
        print("[INFO] Auto-selecting eps using k-distance curve...")

        # Simple method: choose the point where curvature starts rising
        # We use 90th percentile as a starting heuristic
        eps = float(np.percentile(distances, 90))

        print(f"[SUCCESS] Automatically chosen eps: {eps:.3f}")
    else:
        print(f"[INFO] Using manually provided eps = {eps}")

    # ------------------------------------------------------------
    # 3. RUN DBSCAN
    # ------------------------------------------------------------
    print("[INFO] Running DBSCAN...")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    unique_labels = set(labels)
    n_clusters = len([label for label in unique_labels if label != -1])
    n_noise = sum(labels == -1)

    print(f"[INFO] Clusters found (excluding noise): {n_clusters}")
    print(f"[INFO] Noise points: {n_noise}")

    # ------------------------------------------------------------
    # 4. CALCULATE SILHOUETTE SCORE (ignoring noise)
    # ------------------------------------------------------------
    if n_clusters >= 2:
        mask = labels != -1  # ignore noise
        silhouette_value = silhouette_score(X_scaled[mask], labels[mask])
        print(f"[INFO] Silhouette score = {silhouette_value:.4f}")
    else:
        silhouette_value = None
        print("[WARNING] Cannot calculate silhouette (less than 2 clusters).")

    # ------------------------------------------------------------
    # 5. SAVE LABELS
    # ------------------------------------------------------------
    df_output = df_original.copy()
    df_output["DBSCAN_Label"] = labels

    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    df_output.to_csv(labels_path, index=False)

    print(f"[INFO] DBSCAN labels saved to: {labels_path}")

    return eps, labels, silhouette_value



# ------------------------------------------------------------
# TEST BLOCK (runs only when file executed directly)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running dbscan_model.py directly...")

    try:
        from load_data import load_dataset
        from preprocess import preprocess_data

        df_raw, df_numeric = load_dataset()
        _, X_scaled = preprocess_data(df_numeric)

        run_dbscan(X_scaled, df_numeric)

        print("[TEST] dbscan_model.py executed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)


