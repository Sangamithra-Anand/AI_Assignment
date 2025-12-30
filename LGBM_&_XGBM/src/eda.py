# ------------------------------------------------------------
# EDA SCRIPT FOR TITANIC DATASET (FULLY FIXED VERSION)
# ------------------------------------------------------------
# This script performs:
# 1. Loading the Titanic dataset
# 2. Checking missing values
# 3. Generating EDA visualizations
# 4. Automatically creates "output/graphs" folder
# 5. Saves all graphs inside output/graphs/
#
# You run this using: python src/eda.py
# ------------------------------------------------------------

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# FUNCTION 1: Load Titanic Data
# ------------------------------------------------------------
def load_data(train_path, test_path):
    """
    Loads training and test CSV files using pandas.
    """
    print("Loading Dataset...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)

    return train_df, test_df


# ------------------------------------------------------------
# FUNCTION 2: Check Missing Values
# ------------------------------------------------------------
def check_missing_values(df):
    """
    Prints missing values in each column.
    """
    print("\nMissing Values in Dataset:")
    print(df.isnull().sum())


# ------------------------------------------------------------
# HELPER FUNCTION: Save Plots
# ------------------------------------------------------------
def save_plot(fig, filename, output_dir):
    """
    Saves a matplotlib figure inside output_dir.
    This function ensures the directory ALWAYS exists.
    """

    # ----------------------------------------------
    # ⭐ FIX: Create directory inside save_plot also
    # (double protection — folder will 100% exist)
    # ----------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path)
    print(f"Plot saved: {file_path}")
    plt.close(fig)


# ------------------------------------------------------------
# FUNCTION 3: Generate EDA Graphs
# ------------------------------------------------------------
def generate_graphs(df, output_dir):
    """
    Generates all EDA graphs and saves them inside output_dir.
    """
    print("\nGenerating EDA Graphs...")

    # ⭐ FIX: Ensure directory exists BEFORE plotting
    os.makedirs(output_dir, exist_ok=True)

    # -------- 1. Age Distribution --------
    fig = plt.figure(figsize=(7, 5))
    sns.histplot(df["Age"].dropna(), kde=True)
    plt.title("Age Distribution")
    save_plot(fig, "age_distribution.png", output_dir)

    # -------- 2. Fare Distribution --------
    fig = plt.figure(figsize=(7, 5))
    sns.histplot(df["Fare"], kde=True)
    plt.title("Fare Distribution")
    save_plot(fig, "fare_distribution.png", output_dir)

    # -------- 3. Survival Count --------
    fig = plt.figure(figsize=(7, 5))
    sns.countplot(x="Survived", data=df)
    plt.title("Survival Count (0 = No, 1 = Yes)")
    save_plot(fig, "survival_count.png", output_dir)

    # -------- 4. Survival by Gender --------
    fig = plt.figure(figsize=(7, 5))
    sns.countplot(x="Sex", hue="Survived", data=df)
    plt.title("Survival by Gender")
    save_plot(fig, "survival_by_gender.png", output_dir)

    # -------- 5. Fare by Passenger Class --------
    fig = plt.figure(figsize=(7, 5))
    sns.boxplot(x="Pclass", y="Fare", data=df)
    plt.title("Fare by Passenger Class")
    save_plot(fig, "fare_by_class.png", output_dir)

    print("\nAll graphs saved successfully!")


# ------------------------------------------------------------
# MAIN FUNCTION: Controls entire EDA pipeline
# ------------------------------------------------------------
def run_eda():
    """
    Controls the entire EDA workflow.
    """

    # File paths
    train_path = "data/Titanic_train.csv"
    test_path = "data/Titanic_test.csv"

    # Graph output folder
    output_dir = "output/graphs"

    # ⭐ FIX: Create folder BEFORE ANYTHING
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    train_df, test_df = load_data(train_path, test_path)

    # Check missing values
    check_missing_values(train_df)

    # Generate all graphs
    generate_graphs(train_df, output_dir)


# ------------------------------------------------------------
# Run script directly
# ------------------------------------------------------------
if __name__ == "__main__":
    run_eda()
