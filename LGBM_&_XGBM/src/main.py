# ------------------------------------------------------------
# MAIN PIPELINE SCRIPT (Titanic ML Project)
# ------------------------------------------------------------
# This script does EVERYTHING:
# 1. Runs EDA (creates graphs)
# 2. Loads raw dataset
# 3. Preprocesses train & test data
# 4. Trains LightGBM & XGBoost models
# 5. Saves models
# 6. Evaluates models & saves reports
#
# Run using:
#   python src/main.py
# ------------------------------------------------------------

import os
import pandas as pd
from preprocess import preprocess_data
from train_models import train_models
from evaluate import evaluate_models
from eda import run_eda  # ⭐ NEW — Import the EDA function


# ------------------------------------------------------------
# AUTO-CREATE ALL IMPORTANT FOLDERS
# ------------------------------------------------------------
def create_project_folders():
    """
    Ensures all necessary folders are created BEFORE running the pipeline.
    """

    folders = [
        "data",
        "models",
        "output",
        "output/reports",
        "output/graphs"     # ⭐ NEW — Ensure graphs folder exists
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("\nEnsured folders exist:")
    for folder in folders:
        print(f"- {folder}")


# ------------------------------------------------------------
# LOAD RAW DATA (CSV Files)
# ------------------------------------------------------------
def load_raw_data():
    """
    Load Titanic_train.csv and Titanic_test.csv from /data/
    """
    print("\nLoading Raw Data...")

    train_path = "data/Titanic_train.csv"
    test_path  = "data/Titanic_test.csv"

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print(f"Train Data Shape: {train_df.shape}")
    print(f"Test  Data Shape: {test_df.shape}")

    return train_df, test_df


# ------------------------------------------------------------
# MAIN PIPELINE FUNCTION
# ------------------------------------------------------------
def main():

    print("\n================= TITANIC ML PIPELINE STARTED =================")

    # Step 0 → Auto-create folders
    create_project_folders()

    # Step 1 → Run EDA FIRST (creates graphs)
    print("\nRunning EDA...")
    run_eda()
    print("EDA Completed!")

    # Step 2 → Load the raw Titanic dataset
    train_df, test_df = load_raw_data()

    # Step 3 → Preprocess Data
    print("\nPreprocessing Data...")
    X_train, y_train, X_val, y_val, X_test = preprocess_data(train_df, test_df)
    print("Preprocessing Completed!")

    # Step 4 → Train Models
    lgb_model, xgb_model = train_models(X_train, y_train, X_val, y_val)

    # Step 5 → Evaluate Models
    evaluate_models(lgb_model, xgb_model, X_val, y_val)

    print("\n================= PIPELINE COMPLETED SUCCESSFULLY =================")


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()


