"""
main.py
---------------------------------------------------------
This is the MASTER PIPELINE for the entire PCA + Clustering project.

WHAT THIS FILE DOES:
---------------------
Runs ALL steps in the correct order:

    1. Load Dataset
    2. Perform EDA
    3. Preprocess Dataset
    4. Run PCA
    5. Run K-Means on Original Data
    6. Run K-Means on PCA Data
    7. Compare Results
    8. (Optional) Visualizations

This ensures the whole project runs with a single command:
    
        python src/main.py

WHY A MAIN FILE?
----------------
- Makes the entire project runnable from start to finish
- Keeps code modular and clean
- Ideal for your assignment submission or GitHub project
"""

import os
from load_data import load_dataset
from eda import perform_eda
from preprocess import preprocess_data
from pca_model import apply_pca
from clustering_original import run_kmeans_original
from clustering_pca import run_kmeans_pca
from compare_results import compare_clustering_results


# Create main output folders if missing
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/eda_plots", exist_ok=True)
os.makedirs("outputs/clustering_plots", exist_ok=True)
os.makedirs("outputs/pca_plots", exist_ok=True)
os.makedirs("outputs/visuals", exist_ok=True)



# ========================================================================
# MAIN FUNCTION — Runs the entire pipeline
# ========================================================================
def main():
    print("\n======================================================")
    print("       PCA + CLUSTERING PROJECT — FULL PIPELINE")
    print("======================================================\n")

    # ---------------------------------------------------
    # STEP 1 — Load dataset
    # ---------------------------------------------------
    try:
        df_raw = load_dataset()
    except Exception as e:
        print("[FATAL ERROR] Could not load dataset.")
        print(e)
        return

    # ---------------------------------------------------
    # STEP 2 — Perform EDA (histograms, boxplots, heatmap)
    # ---------------------------------------------------
    try:
        perform_eda(df_raw)
    except Exception as e:
        print("[ERROR] EDA failed.")
        print(e)

    # ---------------------------------------------------
    # STEP 3 — Preprocess dataset (numeric-only, scaling)
    # ---------------------------------------------------
    try:
        df_scaled = preprocess_data(df_raw)
    except Exception as e:
        print("[FATAL ERROR] Preprocessing failed.")
        print(e)
        return

    # ---------------------------------------------------
    # STEP 4 — Apply PCA (plots + transform)
    # ---------------------------------------------------
    try:
        df_pca = apply_pca(df_scaled)
    except Exception as e:
        print("[ERROR] PCA failed.")
        print(e)
        return

    # ---------------------------------------------------
    # STEP 5 — K-Means on original scaled data
    # ---------------------------------------------------
    try:
        original_scores = run_kmeans_original(df_scaled)
    except Exception as e:
        print("[ERROR] Clustering on original data failed.")
        print(e)

    # ---------------------------------------------------
    # STEP 6 — K-Means on PCA-transformed data
    # ---------------------------------------------------
    try:
        pca_scores = run_kmeans_pca(df_pca)
    except Exception as e:
        print("[ERROR] Clustering on PCA data failed.")
        print(e)

    # ---------------------------------------------------
    # STEP 7 — Compare results & generate analysis report
    # ---------------------------------------------------
    try:
        compare_clustering_results()
    except Exception as e:
        print("[ERROR] Comparison step failed.")
        print(e)

    print("\n======================================================")
    print("      ✔ Pipeline Completed Successfully!")
    print("      All results saved in the 'outputs/' folder.")
    print("======================================================\n")



# ========================================================================
# EXECUTE ONLY IF RUN DIRECTLY
# ========================================================================
if __name__ == "__main__":
    main()


