"""
main.py
----------------------
This file is the main controller for the entire KNN Zoo Classification Project.

It performs the complete pipeline:
    1. Load dataset
    2. Preprocess data (scaling, checks)
    3. Find best K value (optional)
    4. Train KNN model
    5. Evaluate model performance
    6. Generate visualizations (EDA + decision boundary)

Everything is explained step-by-step inside the code.
"""

# Import modules from src folder
from load_data import load_zoo_data
from preprocess import preprocess_data
from knn_model import train_knn, find_best_k
from evaluate import evaluate_model
from visualize import generate_all_visualizations


# =========================================================
# FUNCTION: main()
# Purpose :
#     - Runs the full machine learning workflow
# =========================================================
def main():
    print("\n============================")
    print(" KNN Zoo Classification App ")
    print("============================\n")

    # ----------------------------------------------
    # Step 1: Load Data
    # ----------------------------------------------
    df = load_zoo_data()

    if df is None:
        print("[FATAL] Dataset not loaded. Exiting program.")
        return

    # ----------------------------------------------
    # Step 2: Preprocess Data (scaling, cleaning)
    # ----------------------------------------------
    df_processed = preprocess_data(df)

    # ----------------------------------------------
    # Step 3 (Optional): Hyperparameter Tuning
    # ----------------------------------------------
    print("\n[INFO] Finding best K value for KNN...")
    best_k = find_best_k(df_processed)
    print(f"[RESULT] Best K selected: {best_k}")

    # ----------------------------------------------
    # Step 4: Train KNN model with best K
    # ----------------------------------------------
    model, X_test, y_test = train_knn(df_processed, k=best_k)

    # ----------------------------------------------
    # Step 5: Evaluate model performance
    # ----------------------------------------------
    print("\n[INFO] Evaluating trained model...")
    metrics = evaluate_model(model, X_test, y_test)

    print("\n[RESULT] Final Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # ----------------------------------------------
    # Step 6: Generate Visualizations
    # ----------------------------------------------
    print("\n[INFO] Generating EDA & Decision Boundary plots...")
    generate_all_visualizations(df_processed)

    print("\n==============================")
    print("   Pipeline Execution DONE    ")
    print("==============================\n")


# =========================================================
# EXECUTION BLOCK
# Runs only when executing:
#     python src/main.py
# =========================================================
if __name__ == "__main__":
    main()


