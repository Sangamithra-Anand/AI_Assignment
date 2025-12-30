import os

# Import all modules from src
from load_data import load_dataset
from preprocess import preprocess_data
from kmeans_model import run_kmeans
from hierarchical_model import run_hierarchical
from dbscan_model import run_dbscan
from visualize import visualize_clusters
from evaluate import evaluate_models


def create_output_folders():
    """
    Creates required output folders if they don't already exist.
    These folders are used by all models to save plots and labels.
    """
    print("[INFO] Checking and creating output folders...")

    os.makedirs("output/plots", exist_ok=True)
    os.makedirs("output/labels", exist_ok=True)
    os.makedirs("output/reports", exist_ok=True)

    print("[INFO] Output folders are ready.")


def main():
    """
    Main controller for the entire Clustering Analysis Project.

    Workflow:
    -------------------------------------------------------------
    1. Create output folders automatically
    2. Load dataset (auto-detect correct sheet)
    3. Preprocess dataset (clean + scale)
    4. Run K-Means clustering
    5. Run Hierarchical clustering
    6. Run DBSCAN clustering
    7. Visualize all cluster results (PCA plots)
    8. Generate evaluation report comparing all models
    """

    print("\n========== CLUSTERING ANALYSIS PROJECT STARTED ==========\n")

    # ---------------------------------------------------------
    # STEP 1: Create folders automatically
    # ---------------------------------------------------------
    create_output_folders()

    # ---------------------------------------------------------
    # STEP 2: Load dataset
    # ---------------------------------------------------------
    print("\n[STEP 2] Loading dataset...")
    df_raw, df_numeric = load_dataset()

    # ---------------------------------------------------------
    # STEP 3: Preprocess dataset
    # ---------------------------------------------------------
    print("\n[STEP 3] Preprocessing dataset...")
    df_clean, X_scaled = preprocess_data(df_numeric)

    # ---------------------------------------------------------
    # STEP 4: Run K-MEANS
    # ---------------------------------------------------------
    print("\n[STEP 4] Running K-Means clustering...")
    kmeans_result = run_kmeans(X_scaled, df_numeric)

    # PCA Visualization for KMeans
    visualize_clusters(
        X_scaled,
        kmeans_result[1],  # labels
        title=f"K-Means Clustering (k={kmeans_result[0]})",
        save_path="output/plots/kmeans_pca_plot.png"
    )

    # ---------------------------------------------------------
    # STEP 5: Run HIERARCHICAL CLUSTERING
    # ---------------------------------------------------------
    print("\n[STEP 5] Running Hierarchical clustering...")
    hierarchical_result = run_hierarchical(X_scaled, df_numeric)

    # PCA Visualization
    visualize_clusters(
        X_scaled,
        hierarchical_result[1],
        title=f"Hierarchical Clustering (clusters={hierarchical_result[0]})",
        save_path="output/plots/hierarchical_pca_plot.png"
    )

    # ---------------------------------------------------------
    # STEP 6: Run DBSCAN
    # ---------------------------------------------------------
    print("\n[STEP 6] Running DBSCAN clustering...")
    dbscan_result = run_dbscan(X_scaled, df_numeric)

    # PCA Visualization
    visualize_clusters(
        X_scaled,
        dbscan_result[1],
        title="DBSCAN Clustering (noise = -1)",
        save_path="output/plots/dbscan_pca_plot.png"
    )

    # ---------------------------------------------------------
    # STEP 7: FINAL EVALUATION REPORT
    # ---------------------------------------------------------
    print("\n[STEP 7] Generating evaluation report...")
    evaluate_models(kmeans_result, hierarchical_result, dbscan_result)

    print("\n========== CLUSTERING ANALYSIS PROJECT COMPLETED ==========\n")


# ---------------------------------------------------------
# RUN MAIN METHOD
# ---------------------------------------------------------
if __name__ == "__main__":
    main()


