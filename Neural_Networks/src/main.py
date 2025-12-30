"""
main.py
--------
This is the MAIN ENTRY POINT of the entire ANN project.

It provides a simple MENU SYSTEM so you can run:

1. Preprocess data
2. Train baseline model
3. Run hyperparameter tuning
4. Evaluate baseline & tuned models
5. Visualize results (loss curves + confusion matrix)
6. Run ALL Steps (Full Pipeline)
7. Exit

Everything is handled in one place for convenience.
"""

import sys

# Import project functions from other modules
from preprocess import preprocess_data, save_processed_data
from data_loader import load_raw_data
from train import train_baseline_model
from tune_hyperparameters import run_hyperparameter_search
from evaluate import evaluate_baseline_and_tuned_models
from visualize_results import main_visualization_pipeline

# From config import create_directories
from config import create_directories


def menu():
    """
    Display the menu options to the user.
    """
    print("""
==========================
      MAIN MENU
==========================
1. Preprocess Data
2. Train Baseline Model
3. Run Hyperparameter Tuning
4. Evaluate Models
5. Visualize Results
6. Run ALL Steps (Full Pipeline)
7. Exit
""")


def run_full_pipeline():
    """
    Runs ALL steps of the project in correct order:

    1. Preprocess data
    2. Train baseline model
    3. Hyperparameter tuning
    4. Evaluate models
    5. Visualize results
    """

    print("\n========== FULL PIPELINE STARTED ==========\n")

    # Step 1 — Preprocess
    print("[1] Preprocessing data...")
    df_raw = load_raw_data()
    df_processed = preprocess_data(df_raw)
    save_processed_data(df_processed)

    # Step 2 — Train baseline
    print("\n[2] Training baseline model...")
    train_baseline_model()

    # Step 3 — Hyperparameter tuning
    print("\n[3] Running hyperparameter tuning...")
    run_hyperparameter_search()

    # Step 4 — Evaluation
    print("\n[4] Evaluating models...")
    evaluate_baseline_and_tuned_models()

    # Step 5 — Visualization
    print("\n[5] Generating visualizations...")
    main_visualization_pipeline()

    print("\n========== FULL PIPELINE COMPLETED ==========\n")


def run_menu():
    """
    Main loop that repeatedly shows the menu
    and runs the selected option.
    """

    # Ensure all required folders exist before doing anything
    create_directories()

    while True:
        menu()
        choice = input("Enter your choice (1-7): ").strip()

        # -----------------------------------------------------------
        # OPTION 1: PREPROCESS DATA
        # -----------------------------------------------------------
        if choice == "1":
            print("\n[INFO] Starting preprocessing...")
            try:
                df_raw = load_raw_data()               # Load raw CSV
                df_processed = preprocess_data(df_raw) # Clean + scale
                save_processed_data(df_processed)      # Save final processed CSV
                print("[INFO] Preprocessing completed successfully.\n")
            except FileNotFoundError as e:
                print(e)

        # -----------------------------------------------------------
        # OPTION 2: TRAIN BASELINE MODEL
        # -----------------------------------------------------------
        elif choice == "2":
            print("\n[INFO] Starting baseline model training...\n")
            try:
                train_baseline_model()
            except FileNotFoundError as e:
                print(e)

        # -----------------------------------------------------------
        # OPTION 3: HYPERPARAMETER TUNING
        # -----------------------------------------------------------
        elif choice == "3":
            print("\n[INFO] Running hyperparameter tuning...\n")
            try:
                run_hyperparameter_search()
            except FileNotFoundError as e:
                print(e)

        # -----------------------------------------------------------
        # OPTION 4: EVALUATE MODELS
        # -----------------------------------------------------------
        elif choice == "4":
            print("\n[INFO] Evaluating baseline and tuned models...\n")
            try:
                evaluate_baseline_and_tuned_models()
            except FileNotFoundError as e:
                print(e)

        # -----------------------------------------------------------
        # OPTION 5: VISUALIZE RESULTS
        # -----------------------------------------------------------
        elif choice == "5":
            print("\n[INFO] Generating visualizations...\n")
            main_visualization_pipeline()

        # -----------------------------------------------------------
        # OPTION 6: RUN FULL PIPELINE
        # -----------------------------------------------------------
        elif choice == "6":
            print("\n[INFO] Running FULL PIPELINE...\n")
            try:
                run_full_pipeline()
            except FileNotFoundError as e:
                print(e)

        # -----------------------------------------------------------
        # OPTION 7: EXIT
        # -----------------------------------------------------------
        elif choice == "7":
            print("\n[INFO] Exiting program. Goodbye!")
            sys.exit(0)

        # -----------------------------------------------------------
        # INVALID OPTION
        # -----------------------------------------------------------
        else:
            print("[ERROR] Invalid choice. Please enter a number from 1 to 7.\n")


# -------------------------------------------------------------------------
# Entry point when running:
#   python main.py
# -------------------------------------------------------------------------
if __name__ == "__main__":
    run_menu()

