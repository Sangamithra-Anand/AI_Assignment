import os

def evaluate_models(
    kmeans_result,
    hierarchical_result,
    dbscan_result,
    save_path="output/reports/clustering_report.txt"
):
    """
    Create a final summary report comparing all clustering models.

    Parameters:
        kmeans_result         → (best_k, kmeans_labels, silhouette)
        hierarchical_result   → (n_clusters, h_labels, silhouette)
        dbscan_result         → (eps_value, db_labels, silhouette)

        save_path             → File path where the report will be saved.

    Report Includes:
    -----------------------------------------------------------
    - Best K of KMeans + silhouette score
    - Best cluster count for Hierarchical + silhouette score
    - EPS used by DBSCAN + silhouette score
    - Total number of clusters detected
    - DBSCAN noise point count
    """

    print("\n[INFO] Creating final evaluation report...")

    # Unpack results
    best_k, kmeans_labels, k_sil = kmeans_result
    h_clusters, h_labels, h_sil = hierarchical_result
    eps_used, db_labels, db_sil = dbscan_result

    # Count noise points for DBSCAN
    noise_count = sum(label == -1 for label in db_labels)

    # Count number of DBSCAN clusters excluding noise
    dbscan_cluster_count = len(set(db_labels) - {-1})

    # -----------------------------------------------------------
    # 1. BUILD REPORT TEXT
    # -----------------------------------------------------------
    report = []
    report.append("CLUSTERING ANALYSIS REPORT")
    report.append("--------------------------------------------\n")

    report.append("1. K-MEANS CLUSTERING")
    report.append(f"   - Best K: {best_k}")
    report.append(f"   - Silhouette Score: {k_sil:.4f}\n")

    report.append("2. HIERARCHICAL CLUSTERING")
    report.append(f"   - Number of Clusters: {h_clusters}")
    report.append(f"   - Silhouette Score: {h_sil:.4f}\n")

    report.append("3. DBSCAN CLUSTERING")
    report.append(f"   - eps used: {eps_used}")
    report.append(f"   - Clusters (excluding noise): {dbscan_cluster_count}")
    report.append(f"   - Noise points: {noise_count}")

    if db_sil is not None:
        report.append(f"   - Silhouette Score: {db_sil:.4f}\n")
    else:
        report.append("   - Silhouette Score: Not applicable (clusters < 2)\n")

    # -----------------------------------------------------------
    # 2. ENSURE DIRECTORY EXISTS
    # -----------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # -----------------------------------------------------------
    # 3. SAVE REPORT
    # -----------------------------------------------------------
    with open(save_path, "w") as f:
        f.write("\n".join(report))

    print(f"[SUCCESS] Clustering report saved to: {save_path}")

    return save_path



# -----------------------------------------------------------
# TEST BLOCK (runs only when executing this file directly)
# -----------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running evaluate.py directly...")

    try:
        from load_data import load_dataset
        from preprocess import preprocess_data
        from kmeans_model import run_kmeans
        from hierarchical_model import run_hierarchical
        from dbscan_model import run_dbscan

        # Load and preprocess
        df_raw, df_numeric = load_dataset()
        _, X_scaled = preprocess_data(df_numeric)

        # Run all three models
        km = run_kmeans(X_scaled, df_numeric)
        hm = run_hierarchical(X_scaled, df_numeric)
        db = run_dbscan(X_scaled, df_numeric)

        # Final evaluation
        evaluate_models(km, hm, db)

        print("[TEST] evaluate.py completed successfully.")

    except Exception as e:
        print("[TEST ERROR]", e)


