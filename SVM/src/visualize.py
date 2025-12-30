# ---------------------------------------------------------------
# VISUALIZATION MODULE
# ---------------------------------------------------------------
# This file contains functions to create and save plots that help
# you understand the mushroom dataset visually.
#
# The functions assume:
#   - the DataFrame may be either raw (categorical text) or encoded
#   - for some plots (like correlation) numeric (encoded) values are required
#   - there is an "output/graphs/" folder in the project root to save plots
#
# Each function accepts a DataFrame and saves the produced figure to
# output/graphs/ with a descriptive filename.
# ---------------------------------------------------------------

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure seaborn default style (clean plots)
sns.set(style="whitegrid")

# Get project root folder correctly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Correct graph folder inside project
GRAPH_DIR = os.path.join(PROJECT_ROOT, "output", "graphs")

# Make sure folder exists
os.makedirs(GRAPH_DIR, exist_ok=True)


def save_fig(fig, filename, dpi=150):
    """
    Save a matplotlib figure to the graphs folder.
    - fig: a matplotlib.figure.Figure object
    - filename: name of the file (string). Example: "class_distribution.png"
    - dpi: image resolution
    """
    path = os.path.join(GRAPH_DIR, filename)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    print(f"Saved plot: {path}")


def plot_class_distribution(df, save=True):
    """
    Plot bar/count plot for class distribution (edible vs poisonous).
    - df: DataFrame which must include a 'class' column (encoded or raw)
    - save: if True, saves the plot to GRAPH_DIR
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    # If class is encoded as numbers, countplot still works
    sns.countplot(x='class', data=df, ax=ax)
    ax.set_title("Class Distribution (edible vs poisonous)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    if save:
        save_fig(fig, "class_distribution.png")
    plt.close(fig)   # close the figure to free memory


def plot_categorical_vs_class(df, column, rotate_x=False, save=True):
    """
    Plot a categorical column vs class as a countplot.
    - column: string, column name to visualize (e.g., 'odor', 'habitat')
    - rotate_x: rotate x labels if they overlap
    - df: the DataFrame with the data (can be encoded or raw)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x=column, hue='class', data=df, ax=ax)
    ax.set_title(f"{column} vs Class")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")

    if rotate_x:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    if save:
        safe_name = f"{column}_vs_class.png"
        save_fig(fig, safe_name)
    plt.close(fig)


def plot_correlation_matrix(encoded_df, save=True):
    """
    Plot the correlation matrix heatmap.
    NOTE: This function expects a numeric (encoded) dataframe.
    - encoded_df: DataFrame where columns are numeric
    - save: whether to save the figure
    """
    # Compute the Pearson correlation matrix
    corr = encoded_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    # Use a diverging palette centered at 0
    sns.heatmap(corr, annot=False, cmap="vlag", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")

    if save:
        save_fig(fig, "correlation_matrix.png")
    plt.close(fig)


def plot_top_correlated_features(encoded_df, target_col='class', top_n=10, save=True):
    """
    Show the top_n features most correlated with the target column.
    - encoded_df: numeric dataframe (encoded)
    - target_col: the name of the label column (default 'class')
    - top_n: how many top positive/negative correlated features to show
    """
    if target_col not in encoded_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # Compute correlation of all features with the target column
    corr_target = encoded_df.corr()[target_col].drop(target_col).sort_values(ascending=False)

    # Take top positive and top negative correlations, then combine
    top_positive = corr_target.head(top_n)
    top_negative = corr_target.tail(top_n).sort_values()
    combined = pd.concat([top_positive, top_negative])

    fig, ax = plt.subplots(figsize=(8, 6))
    combined.plot(kind='barh', ax=ax)
    ax.set_title(f"Top correlated features with '{target_col}'")
    ax.set_xlabel("Pearson correlation")

    if save:
        save_fig(fig, f"top_correlated_with_{target_col}.png")
    plt.close(fig)
