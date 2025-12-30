import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def run_hierarchical(
    X_scaled,
    df_original,
    dendrogram_path="output/plots/dendrogram.png",
    labels_path="output/labels/hierarchical_labels.csv",
    n_clusters=None
):
    """
    Performs Hierarchical Agglomerative Clustering.

    Steps:
    ------------------------------------------------------------
    1. Generate a dendrogram using Ward linkage
    2. Save the dendrogram as a PNG image
    3. If n_clusters is NOT given → automatically estimate clusters
    4. Train AgglomerativeClustering
    5. Compute silhouette score
    6. Save output labels to CSV

    Args:
        X_scaled        → Scaled numeric data (numpy array or DataFrame)
        df_original     → Original dataset (for saving cluster labels)
        dendrogram_path → Path to save dendrogram image
        labels_path     → Path to save cluster labels
        n_clusters      → Optional manual number of clusters

    Returns:
        n_clusters_used → Number of clusters used
        labels          → Cluster label array
        silhouette      → Silhouette score
    """

    print("\n[INFO] Starting Hierarchical Clustering...")

    # ------------------------------------------------------------
    # 1. CREATE DENDROGRAM
    # ------------------------------------------------------------
    print("[INFO] Generating dendrogram using Ward linkage...")

    # Using only first 200 rows for clean visualization
    sample_data = X_scaled[:200]

    Z = linkage(sample_data, method="ward")

    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="lastp", p=30)
    plt.title("Hierarchical Clustering Dendrogram (Truncated)")
    plt.xlabel("Cluster Size")
    plt.ylabel("Distance")

    # Ensure directory exists
    os.makedirs(os.path.dirname(dendrogram_path), exist_ok=True)

    plt.savefig(dendrogram_path)
    plt.close()

    print(f"[INFO] Dendrogram saved to: {dendrogram_path}")

    # ------------------------------------------------------------
    # 2. DETERMINE NUMBER OF CLUSTERS (if not given)
    # ------------------------------------------------------------
    if n_clusters is None:
        print("[INFO] Automatically selecting cluster count...")

        # Try silhouette for k = 2 to 10, similar to KMeans logic
        sil_scores = {}
        for k in range(2, 11):
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = model.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            sil_scores[k] = sil
            print(f"[INFO] k={k} → silhouette={sil:.4f}")

        n_clusters = max(sil_scores, key=sil_scores.get)
        print(f"[SUCCESS] Best cluster count chosen: {n_clusters}")

    else:
        print(f"[INFO] Using manually provided number of clusters: {n_clusters}")

    # ------------------------------------------------------------
    # 3. RUN FINAL AGGLOMERATIVE CLUSTERING
    # ------------------------------------------------------------
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X_scaled)

    print("[INFO] Hierarchical clustering completed.")

    # ------------------------------------------------------------
    # 4. COMPUTE SILHOUETTE SCORE
    # ------------------------------------------------------------
    silhouette = silhouette_score(X_scaled, labels)
    print(f"[INFO] Silhouette score = {silhouette:.4f}")

    # ------------------------------------------------------------
    # 5. SAVE LABELS TO CSV
    # ------------------------------------------------------------
    df_output = df_original.copy()
    df_output["Hierarchical_Label"] = labels

    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    df_output.to_csv(labels_path, index=False)

    print(f"[INFO] Cluster labels saved to: {labels_path}")

    return n_clusters, labels, silhouette



# ------------------------------------------------------------
# TEST BLOCK (runs only if executing this file directly)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running hierarchical_model.py directly...")

    try:
        from load_data import load_dataset
        from preprocess import preprocess_data

        # Load and preprocess
        df_raw, df_numeric = load_dataset()
        _, X_scaled = preprocess_data(df_numeric)

        # Run hierarchical clustering
        run_hierarchical(X_scaled, df_numeric)

        print("[TEST] hierarchical_model.py completed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)

