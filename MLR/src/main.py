"""
main.py
--------
This is the MAIN PIPELINE CONTROLLER for the Toyota Corolla MLR Project.

It connects all modules:
1. EDA
2. Preprocessing
3. Training 3 Regression Models
4. Evaluating models
5. Lasso & Ridge Regularization

This file provides a simple MENU-BASED interface so the user can run each
step independently or run the FULL PIPELINE.
"""

import os
import pandas as pd

# Import all pipeline components
from eda import run_eda
from preprocess import preprocess_data
from train_models import train_models
from evaluate import evaluate_all_models
from regularization import train_regularization_models
from utils import load_cleaned_data


# =====================================================================================
# Helper Function: load_raw_dataset()
# =====================================================================================
def load_raw_dataset(path="data/raw/ToyotaCorolla - MLR.csv"):
    """
    Loads the raw dataset from the data/raw folder.
    Returns DataFrame or None (if file missing).
    """
    if not os.path.exists(path):
        print(f"[ERROR] Raw dataset not found at: {path}")
        return None

    print(f"[INFO] Loading raw dataset from: {path}")
    return pd.read_csv(path)


# =====================================================================================
# FULL PIPELINE RUNNER
# =====================================================================================
def run_full_pipeline():
    """
    Runs the entire ML pipeline:
    EDA → Preprocessing → Train Models → Evaluation → Lasso & Ridge
    """

    print("\n===== RUNNING FULL MACHINE LEARNING PIPELINE =====")

    # Step 1: Load Raw Dataset
    df_raw = load_raw_dataset()
    if df_raw is None:
        return

    # Step 2: EDA
    run_eda(df_raw)

    # Step 3: Preprocessing
    cleaned_df = preprocess_data(df_raw)

    # Step 4: Train Models
    train_models(cleaned_df)

    # Step 5: Evaluate Models
    evaluate_all_models(cleaned_df)

    # Step 6: Lasso & Ridge Regularization
    train_regularization_models(cleaned_df)

    print("\n===== FULL PIPELINE COMPLETED SUCCESSFULLY =====")


# =====================================================================================
# MENU SYSTEM (FIXED)
# =====================================================================================
def show_menu():
    """
    Displays the menu and handles user input.
    This version is FIXED to prevent premature exit.
    """
    while True:
        print("\n========== TOYOTA COROLLA MLR PROJECT ==========")
        print("1. Run EDA (Exploratory Data Analysis)")
        print("2. Run Preprocessing")
        print("3. Train Regression Models")
        print("4. Evaluate Models")
        print("5. Run Lasso & Ridge Regularization")
        print("\033[1;33m6. Run FULL PIPELINE (RECOMMENDED) ⭐\033[0m")
        print("7. Exit")
        print("==============================================")

        try:
            choice = input("Enter your choice (1-7): ")
        except KeyboardInterrupt:
            print("\n[INFO] Exiting program safely. Goodbye!")
            break  # exit menu loop

        # ----------------------------------------------------------
        # Load raw & cleaned datasets only when needed
        # ----------------------------------------------------------
        df_raw = None
        cleaned_df = None

        if choice in ["1", "2", "6"]:
            df_raw = load_raw_dataset()

        if choice in ["3", "4", "5", "6"]:
            cleaned_df = load_cleaned_data()

        # ----------------------------------------------------------
        # MENU HANDLER
        # ----------------------------------------------------------
        if choice == "1":
            if df_raw is not None:
                run_eda(df_raw)

        elif choice == "2":
            if df_raw is not None:
                preprocess_data(df_raw)

        elif choice == "3":
            if cleaned_df is not None:
                train_models(cleaned_df)

        elif choice == "4":
            if cleaned_df is not None:
                evaluate_all_models(cleaned_df)

        elif choice == "5":
            if cleaned_df is not None:
                train_regularization_models(cleaned_df)

        elif choice == "6":
            run_full_pipeline()

        elif choice == "7":
            print("\n[INFO] Exiting program. Goodbye!")
            break  # SUCCESSFULL EXIT

        else:
            print("[ERROR] Invalid choice. Please select a number (1–7).")


# =====================================================================================
# DIRECT EXECUTION
# =====================================================================================
if __name__ == "__main__":
    show_menu()
