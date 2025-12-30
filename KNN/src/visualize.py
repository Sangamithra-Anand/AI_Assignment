"""
visualize.py
----------------------
This module creates all visualizations used in the KNN Zoo Classification project.

It includes:
    ✔ Distribution plots for each numeric feature
    ✔ Correlation heatmap (fixed → uses only numeric columns)
    ✔ KNN decision boundary visualization (2D)

Every function contains detailed explanations inside the code.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# =========================================================
# FUNCTION: save_plot()
# Purpose:
#   Save a plot into the output/plots/ folder.
#   - Automatically creates the folder if missing
#   - Ensures clean, organized outputs
# =========================================================
def save_plot(filename):
    os.makedirs("output/plots", exist_ok=True)  # Folder created if missing
    path = f"output/plots/{filename}"

    plt.savefig(path, dpi=300, bbox_inches="tight")  # High-quality image
    print(f"[INFO] Plot saved: {path}")


# =========================================================
# FUNCTION: plot_distributions()
# Purpose:
#   For every numeric column in the dataset:
#       • Create a histogram (distribution plot)
#       • Add KDE curve for better smoothing
#       • Save each plot individually
#
#   This helps us understand how each feature is distributed.
# =========================================================
def plot_distributions(df):
    print("\n[INFO] Creating distribution plots...")

    # Select numeric columns only for plotting
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Loop over numeric columns and generate histogram per column
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))

        # histplot = histogram + optional KDE for smooth density
        sns.histplot(df[col], kde=True)

        plt.title(f"Distribution of {col}")  # Graph title
        plt.xlabel(col)
        plt.ylabel("Count")

        # Save each plot with its column name
        save_plot(f"dist_{col}.png")
        plt.close()  # Close figure to avoid memory buildup


# =========================================================
# FUNCTION: plot_correlation_heatmap()
# Purpose:
#   - Shows correlation between numeric features.
#   - Correlation = how much two variables change together.
#   - Only NUMERIC columns allowed (fixes string errors).
#
# Why numeric only?
#   Pandas cannot compute correlation for strings like "aardvark".
# =========================================================
def plot_correlation_heatmap(df):
    print("\n[INFO] Generating correlation heatmap...")

    # Keep numeric columns only — FIX for "could not convert to float" error
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    plt.figure(figsize=(10, 8))

    # Compute correlation matrix
    corr = numeric_df.corr()

    # Heatmap visualization with color gradients
    sns.heatmap(
        corr,
        annot=False,       # We disable text annotations for cleaner look
        cmap="coolwarm",   # Red-blue gradient
        linewidths=0.5     # Separation lines
    )

    plt.title("Feature Correlation Heatmap (Numeric Only)")
    save_plot("correlation_heatmap.png")
    plt.close()


# =========================================================
# FUNCTION: plot_decision_boundary()
# Purpose:
#   Visualize how KNN separates animal types in a 2D feature space.
#
#   Steps:
#     1. Take two numeric features (feature1, feature2)
#     2. Train a KNN model using ONLY these 2 columns
#     3. Generate a large grid of points
#     4. Predict class for every point → forms decision regions
#     5. Plot class regions + actual data points
#
# Why only 2 features?
#   Decision boundaries can only be visualized in 2D.
# =========================================================
def plot_decision_boundary(df, feature1, feature2, target_column="type", k=5):
    print(f"\n[INFO] Creating decision boundary for: {feature1} vs {feature2}")

    # Extract required columns into NumPy arrays
    X = df[[feature1, feature2]].values  # 2D feature matrix
    y = df[target_column].values          # Labels

    # Create a KNN model using chosen K value
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)  # Train KNN on 2D data

    # ---------------------------------------------------------
    # Create a mesh grid covering the full 2D range of features
    # ---------------------------------------------------------
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a smooth grid with 400 x 400 points
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    # Flatten grid and predict class for each point
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points).reshape(xx.shape)

    # ---------------------------------------------------------
    # Plot the decision boundary regions
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))

    # Filled contour map showing predicted classes
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="viridis")

    # Plot actual data points on top
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f"KNN Decision Boundary ({feature1} vs {feature2})")

    # Add legend for animal types
    plt.legend(*scatter.legend_elements(), title="Animal Type")

    filename = f"decision_boundary_{feature1}_{feature2}.png"
    save_plot(filename)
    plt.close()


# =========================================================
# FUNCTION: generate_all_visualizations()
# Purpose:
#   Run ALL visualization functions in a clean, organized pipeline.
#
# Called from main.py at the end of training.
# =========================================================
def generate_all_visualizations(df):
    print("\n[INFO] Running visualization pipeline...")

    # Step 1: Distribution plots
    plot_distributions(df)

    # Step 2: Correlation map
    plot_correlation_heatmap(df)

    # Step 3: Decision boundary example
    # You may choose any two numeric features.
    plot_decision_boundary(
        df,
        feature1="legs",
        feature2="hair",
        target_column="type",
        k=5
    )

    print("[INFO] Visualization pipeline completed.")


# =========================================================
# TEST BLOCK (runs only when executing this file directly)
# ---------------------------------------------------------
# Command:
#   python src/visualize.py
#
# Useful for checking:
#   • plot functions
#   • missing folder issues
#   • error debugging
# =========================================================
if __name__ == "__main__":
    import pandas as pd

    print("[TEST] Testing visualize.py...")

    try:
        df_sample = pd.read_csv("data/Zoo.csv")
        generate_all_visualizations(df_sample)

        print("[TEST] Visualization test completed successfully.")

    except Exception as e:
        print("[TEST ERROR] Visualization test failed:", e)


