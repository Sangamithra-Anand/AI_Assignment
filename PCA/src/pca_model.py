"""
pca_model.py
---------------------------------------------------------
This file performs Principal Component Analysis (PCA) on the
preprocessed (scaled) dataset.

WHAT THIS FILE DOES:
---------------------
1. Loads scaled dataset from preprocess.py output
2. Fits PCA on numeric features
3. Generates:
      ✔ Scree plot (explained variance per component)
      ✔ Cumulative explained variance plot
4. Transforms dataset into PCA components
5. Saves:
      ✔ PCA-transformed dataset
      ✔ PCA variance summary (JSON)
6. Returns the PCA-transformed DataFrame

WHY PCA IS NEEDED:
------------------
- Reduces dimensionality while keeping maximum variance
- Removes multicollinearity (high correlation between features)
- Helps clustering perform better
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Output folders for PCA plots + reports
PCA_PLOTS_DIR = "outputs/pca_plots"
REPORTS_DIR = "outputs/reports"
PROCESSED_DIR = "data/processed"

os.makedirs(PCA_PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)



# ========================================================================
# FUNCTION: apply_pca(df, n_components=None)
# PURPOSE :
#   Runs PCA on the dataset, generates visualizations,
#   and saves PCA-transformed data.
# ========================================================================
def apply_pca(df, n_components=None):
    """
    Performs PCA on the scaled dataset.

    Args:
        df (pandas.DataFrame): Scaled numeric dataframe
        n_components (int or None): Number of components to keep.
                                    If None → keep all components.

    Returns:
        pandas.DataFrame: PCA-transformed dataset

    STEPS:
        ✔ Fit PCA
        ✔ Plot explained variance
        ✔ Save PCA components
        ✔ Return new PCA DataFrame
    """

    print("\n[INFO] Starting PCA analysis...")

    # -------------------------------------------------------------
    # Step 1: Initialize PCA model
    #
    # n_components = None → PCA automatically keeps all components.
    #
    # PCA learns:
    #   - PC1 (most variance)
    #   - PC2 (2nd most variance)
    #   - ...
    # -------------------------------------------------------------
    print("[INFO] Fitting PCA model...")

    pca = PCA(n_components=n_components)
    pca.fit(df)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print("[INFO] PCA model fitted successfully.")


    # -------------------------------------------------------------
    # Step 2: Create Scree Plot
    #
    # A scree plot shows how much variance each PCA component explains.
    # Helps decide optimal number of components.
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1),
             explained_variance,
             marker='o')

    plt.title("Scree Plot - Explained Variance per Component")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")

    scree_path = f"{PCA_PLOTS_DIR}/scree_plot.png"
    plt.savefig(scree_path)
    plt.close()

    print(f"[INFO] Scree plot saved at: {scree_path}")


    # -------------------------------------------------------------
    # Step 3: Cumulative Explained Variance Plot
    #
    # Helps answer:
    #   "How many components are needed to retain 90–95% variance?"
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance,
             marker='o')

    plt.axhline(y=0.90, color='r', linestyle='--', label="90% Variance")
    plt.axhline(y=0.95, color='g', linestyle='--', label="95% Variance")

    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.legend()

    cum_var_path = f"{PCA_PLOTS_DIR}/cumulative_variance.png"
    plt.savefig(cum_var_path)
    plt.close()

    print(f"[INFO] Cumulative variance plot saved at: {cum_var_path}")


    # -------------------------------------------------------------
    # Step 4: Save PCA explained variance summary
    # -------------------------------------------------------------
    summary_path = f"{REPORTS_DIR}/pca_summary.json"

    summary_data = {
        "explained_variance_ratio": explained_variance.tolist(),
        "cumulative_variance_ratio": cumulative_variance.tolist(),
        "recommended_components_90": int(np.argmax(cumulative_variance >= 0.90) + 1),
        "recommended_components_95": int(np.argmax(cumulative_variance >= 0.95) + 1)
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"[INFO] PCA summary saved at: {summary_path}")


    # -------------------------------------------------------------
    # Step 5: Transform dataset into PCA components
    # -------------------------------------------------------------
    print("[INFO] Transforming dataset into PCA components...")

    pca_data = pca.transform(df)

    # Convert to DataFrame
    pca_columns = [f"PC{i}" for i in range(1, pca_data.shape[1] + 1)]
    pca_df = pd.DataFrame(pca_data, columns=pca_columns)

    print("[INFO] PCA transformation completed.")


    # -------------------------------------------------------------
    # Step 6: Save PCA-transformed dataset
    # -------------------------------------------------------------
    save_path = f"{PROCESSED_DIR}/pca_transformed.csv"
    pca_df.to_csv(save_path, index=False)

    print(f"[INFO] PCA-transformed dataset saved at: {save_path}")


    print("\n[INFO] PCA analysis completed successfully!\n")
    return pca_df



# ========================================================================
# TESTING BLOCK – runs only when file is executed directly
# Lets you test PCA before using main.py
# ========================================================================
if __name__ == "__main__":
    from load_data import load_dataset
    from preprocess import preprocess_data

    print("[TEST] Running pca_model.py directly...")

    try:
        df = load_dataset()
        df_scaled = preprocess_data(df)
        pca_df = apply_pca(df_scaled)

        print("[TEST] pca_model.py executed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)

