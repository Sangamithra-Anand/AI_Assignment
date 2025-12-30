import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_kmeans(X_scaled, df_original, save_plot_path="output/plots/elbow_plot.png",
               save_labels_path="output/labels/kmeans_labels.csv"):
    """
    Performs K-Means clustering on scaled data.

    Steps performed:
    -----------------------------------------------------------
    1. Compute inertia values for K = 1 to 10  (Elbow Method)
    2. Plot Elbow Curve and save it
    3. Compute Silhouette scores for K = 2 to 10
    4. Select best K using silhouette (highest score)
    5. Run final K-Means model with best K
    6. Save cluster labels in output/labels folder

    Args:
        X_scaled       → scaled numeric data (DataFrame)
        df_original    → original unscaled numeric data
        save_plot_path → where elbow plot will be saved
        save_labels_path → where cluster labels will be saved

    Returns:
        best_k       → optimal number of clusters
        labels       → array of cluster labels for each row
        silhouette   → silhouette score for best_k
    """

    print("\n[INFO] Starting K-Means clustering...")

    # -----------------------------------------------------------
    # 1. COMPUTE INERTIA FOR ELBOW METHOD
    # -----------------------------------------------------------
    inertia = []
    K_range = range(1, 11)

    print("[INFO] Computing inertia values for K = 1 to 10...")

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # -----------------------------------------------------------
    # 2. PLOT ELBOW CURVE
    # -----------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(K_range, inertia, marker='o')
    plt.title("Elbow Curve for K-Means")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(True)

    # Create folder if missing
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)

    plt.savefig(save_plot_path)
    plt.close()
    print(f"[INFO] Elbow plot saved to: {save_plot_path}")

    # -----------------------------------------------------------
    # 3. COMPUTE SILHOUETTE SCORES
    # -----------------------------------------------------------
    print("[INFO] Calculating silhouette scores for K = 2 to 10...")

    sil_scores = {}
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        sil_scores[k] = sil
        print(f"[INFO] K={k} → silhouette={sil:.4f}")

    # -----------------------------------------------------------
    # 4. SELECT BEST K BASED ON HIGHEST SILHOUETTE SCORE
    # -----------------------------------------------------------
    best_k = max(sil_scores, key=sil_scores.get)
    best_silhouette = sil_scores[best_k]

    print(f"[SUCCESS] Best K selected: {best_k} (silhouette={best_silhouette:.4f})")

    # -----------------------------------------------------------
    # 5. RUN FINAL KMEANS MODEL USING BEST K
    # -----------------------------------------------------------
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_model.fit_predict(X_scaled)

    print("[INFO] Final K-Means model trained.")

    # -----------------------------------------------------------
    # 6. SAVE LABELS TO output/labels
    # -----------------------------------------------------------
    df_output = df_original.copy()
    df_output["KMeans_Label"] = labels

    os.makedirs(os.path.dirname(save_labels_path), exist_ok=True)
    df_output.to_csv(save_labels_path, index=False)

    print(f"[INFO] K-Means labels saved to: {save_labels_path}")

    return best_k, labels, best_silhouette



# -----------------------------------------------------------
# TEST BLOCK (runs only when you execute this file directly)
# -----------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running kmeans_model.py directly...")

    try:
        from load_data import load_dataset
        from preprocess import preprocess_data

        df_raw, df_numeric = load_dataset()
        _, X_scaled = preprocess_data(df_numeric)

        run_kmeans(X_scaled, df_numeric)

        print("[TEST] kmeans_model.py executed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)


