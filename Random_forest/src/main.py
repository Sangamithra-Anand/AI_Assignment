"""
main.py
--------
This is the MASTER PIPELINE CONTROLLER for the Glass Classification Project.

It performs the following tasks:

1. Load raw data
2. Run Exploratory Data Analysis (EDA)
3. Generate visualizations (histograms, boxplots, heatmaps)
4. Preprocess data (scaling, SMOTE, cleaning)
5. Train Random Forest model
6. Train Bagging & Boosting models
7. Display final summary

The goal:
---------
- Give the user a clean, step-by-step executable program.
- No Jupyter Notebook needed.
"""

import pandas as pd
from load_data import load_glass_data
from eda import run_eda
from visualize import run_visualizations
from preprocess import preprocess_glass_data
from train_random_forest import train_random_forest
from bagging_boosting import run_bagging_and_boosting

# utils helpers
from utils import log_message, start_timer, end_timer


# -------------------------------------------------------------------------
# Function: Run FULL pipeline
# -------------------------------------------------------------------------
def run_full_pipeline():
    """
    Runs ALL steps in sequence without a menu loop.

    Steps:
    ------
    1. Load raw dataset
    2. Run EDA
    3. Generate visualizations
    4. Preprocess dataset
    5. Train Random Forest model
    6. Train Bagging & Boosting models

    Notes:
    ------
    - This function is the heart of the project.
    - It logs progress and measures execution time.
    """

    log_message("FULL PIPELINE STARTED")
    timer = start_timer()

    # -----------------------------
    # STEP 1: Load raw data
    # -----------------------------
    log_message("Loading raw dataset...")
    df_raw = load_glass_data()

    # -----------------------------
    # STEP 2: Run EDA
    # -----------------------------
    log_message("Running EDA...")
    run_eda(df_raw)

    # -----------------------------
    # STEP 3: Generate visualizations
    # -----------------------------
    log_message("Generating visualizations...")
    run_visualizations(df_raw)

    # -----------------------------
    # STEP 4: Preprocess dataset
    # -----------------------------
    log_message("Preprocessing dataset...")
    df_cleaned = preprocess_glass_data(df_raw)

    # -----------------------------
    # STEP 5: Train Random Forest
    # -----------------------------
    log_message("Training Random Forest model...")
    rf_model, rf_metrics = train_random_forest(df_cleaned)

    # -----------------------------
    # STEP 6: Train Bagging + Boosting models
    # -----------------------------
    log_message("Training Bagging and Boosting models...")
    bagging_metrics, boosting_metrics = run_bagging_and_boosting(df_cleaned)

    # -----------------------------
    # FINAL SUMMARY
    # -----------------------------
    print("\n================ FINAL RESULTS SUMMARY ================\n")
    print("Random Forest Metrics:")
    print(rf_metrics)
    print("\nBagging Metrics:")
    print(bagging_metrics)
    print("\nBoosting Metrics:")
    print(boosting_metrics)
    print("\n=======================================================\n")

    log_message("FULL PIPELINE COMPLETED")
    end_timer(timer, "Full Pipeline Execution")


# -------------------------------------------------------------------------
# RUN MAIN PIPELINE
# -------------------------------------------------------------------------
if __name__ == "__main__":
    """
    When you run:
        python src/main.py

    The FULL ML pipeline will run from start to finish.
    No menu is used because you want continuous execution.
    """

    print("\n====================================================")
    print(" GLASS CLASSIFICATION PROJECT — FULL ML PIPELINE")
    print("====================================================")

    run_full_pipeline()

    print("\n🎉 Pipeline completed successfully! Check your outputs folder, models, logs, and reports.")
