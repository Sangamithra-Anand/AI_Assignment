import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_clusters(
    X_scaled,
    labels,
    title,
    save_path="output/plots/cluster_plot.png"
):
    """
    Visualizes clustering results using PCA (2D).

    Why PCA?
    --------
    Most datasets have many columns.
    PCA reduces them to 2D so we can visualize clusters.

    Parameters:
        X_scaled  → Scaled numeric data (np.array or DataFrame)
        labels    → Cluster labels (KMeans / Hierarchical / DBSCAN)
        title     → Title of the plot (string)
        save_path → Where to save the image file

    Returns:
        None (Saves PNG image)
    """

    print(f"\n[INFO] Creating PCA visualization: {title}")

    # -----------------------------------------------------------
    # 1. PCA REDUCTION TO 2D
    # -----------------------------------------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print("[INFO] PCA completed. Variance explained:",
          pca.explained_variance_ratio_.sum())

    # -----------------------------------------------------------
    # 2. PLOT USING MATPLOTLIB
    # -----------------------------------------------------------
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis")

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Optional legend for cluster colors
    if len(set(labels)) < 15:  # avoid too large legend
        plt.legend(*scatter.legend_elements(), title="Clusters")

    # -----------------------------------------------------------
    # 3. SAVE PLOT
    # -----------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Plot saved to: {save_path}")


# -----------------------------------------------------------
# TEST BLOCK (runs only when executed directly)
# -----------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running visualize.py directly...")

    try:
        from load_data import load_dataset
        from preprocess import preprocess_data
        from kmeans_model import run_kmeans

        # Load & preprocess
        df_raw, df_numeric = load_dataset()
        df_clean, X_scaled = preprocess_data(df_numeric)

        # Run KMeans for test visualization
        best_k, labels, sil = run_kmeans(X_scaled, df_numeric)

        # Visualize
        visualize_clusters(
            X_scaled,
            labels,
            title=f"K-Means Visualization (k={best_k})",
            save_path="output/plots/kmeans_pca_plot.png"
        )

        print("[TEST] visualize.py executed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)


