import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# -------------------------------------------------------------
# Function: plot_correlation_matrix
# Purpose : Create and save a heatmap of the correlation matrix.
# -------------------------------------------------------------
def plot_correlation_matrix(df, save_path="output/correlation_matrix.png"):
    """
    Generates a correlation matrix heatmap for numerical features.

    Correlation:
    - Shows linear relationships between numeric variables
    - Helps identify multicollinearity
    """

    print("[INFO] Creating correlation matrix heatmap...")

    # Get only numeric features for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    corr_matrix = numeric_df.corr()

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
    plt.title("Correlation Matrix Heatmap")

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[INFO] Correlation matrix saved to: {save_path}")


# -------------------------------------------------------------
# Function: compute_mutual_information_matrix
# Purpose : Compute MI values for all feature pairs.
# NOTE    : This is NOT the same as PPS but serves as a good
#           non-linear relevance visualizer.
# -------------------------------------------------------------
def compute_mutual_information_matrix(df):
    """
    Computes a mutual information matrix for all numeric features.

    MI tells how much information one variable gives about another.

    Returns:
        mi_matrix (DataFrame): MI score table for visualization
    """

    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    columns = numeric_df.columns
    mi_matrix = pd.DataFrame(index=columns, columns=columns)

    print("[INFO] Computing Mutual Information matrix...")

    # Compute MI for each pair of features
    for col_x in columns:
        for col_y in columns:

            # MI expects the target as 1D array
            x = numeric_df[col_x].values.reshape(-1, 1)
            y = numeric_df[col_y].values

            # Regression MI (numeric → numeric)
            mi_value = mutual_info_regression(x, y, discrete_features=False)

            mi_matrix.loc[col_x, col_y] = mi_value[0]

    # Convert to numeric (float)
    mi_matrix = mi_matrix.astype(float)

    return mi_matrix


# -------------------------------------------------------------
# Function: plot_mutual_information_heatmap
# Purpose : Visualize MI matrix in heatmap form.
# -------------------------------------------------------------
def plot_mutual_information_heatmap(df, save_path="output/mutual_information_heatmap.png"):
    """
    Creates a heatmap of mutual information between all numeric features.
    This shows how strongly each feature relates to others.
    """

    mi_matrix = compute_mutual_information_matrix(df)

    os.makedirs("output", exist_ok=True)

    plt.figure(figsize=(14, 12))
    sns.heatmap(mi_matrix, cmap="Blues", annot=False)
    plt.title("Mutual Information Heatmap")

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[INFO] MI heatmap saved to: {save_path}")


# -------------------------------------------------------------
# Function: visualization_pipeline
# Purpose : Run correlation + MI heatmap visualizations.
# -------------------------------------------------------------
def visualization_pipeline(df):
    """
    Runs all visualizations together:
    1. Correlation matrix heatmap
    2. Mutual Information heatmap
    """

    print("[INFO] Starting visualization pipeline...")

    plot_correlation_matrix(df)
    plot_mutual_information_heatmap(df)

    print("[INFO] Visualization pipeline completed.")
