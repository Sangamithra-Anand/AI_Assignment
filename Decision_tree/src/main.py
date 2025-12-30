"""
main.py
--------
This is the MAIN PIPELINE CONTROLLER for the Decision Tree Project.

What this script does:
1. Auto-creates all required folders (via utils.py)
2. Loads the dataset
3. Preprocesses the dataset
4. Runs Feature Engineering
5. Performs EDA
6. Trains Decision Tree model
7. Evaluates model performance
8. Visualizes the Decision Tree
9. Logs every major step

Every important process is explained inside this file.
"""

from utils import create_project_folders, log_message, start_timer, end_timer
from load_data import load_raw_dataset
from preprocess import preprocess_data
from feature_engineering import feature_engineering
from eda import run_eda
from train_model import train_decision_tree
from evaluate import evaluate_model
from visualize_tree import visualize_tree


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():
    """Runs the full machine learning workflow step-by-step."""

    print("\n====================================================")
    print("      DECISION TREE CLASSIFICATION PIPELINE")
    print("====================================================")

    # ----------------------------------------------------
    # 1. Auto-create folders
    # ----------------------------------------------------
    create_project_folders()
    log_message("[INFO] Project folders created.")

    # ----------------------------------------------------
    # 2. Load Raw Dataset
    # ----------------------------------------------------
    load_timer = start_timer()
    df_raw = load_raw_dataset()  # Reads data/raw/heart_disease.xlsx
    end_timer(load_timer, "Loading Raw Dataset")

    if df_raw is None:
        print("[FATAL] Dataset could not be loaded. Exiting pipeline.")
        log_message("[FATAL] Dataset load failed. Pipeline stopped.")
        return

    # ----------------------------------------------------
    # 3. Preprocess Data
    # ----------------------------------------------------
    preprocess_timer = start_timer()
    df_clean = preprocess_data(df_raw)  # Save cleaned data
    end_timer(preprocess_timer, "Preprocessing Dataset")

    # ----------------------------------------------------
    # 4. Feature Engineering
    # Update the columns based on your dataset
    # ----------------------------------------------------
    fe_timer = start_timer()

    label_encode_cols = ["sex", "fbs", "exang"]
    one_hot_cols = ["cp", "restecg", "slope", "thal"]
    scale_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    df_final = feature_engineering(
        df_clean,
        label_encode_cols=label_encode_cols,
        one_hot_cols=one_hot_cols,
        scale_cols=scale_cols,
        scale_method="standard"
    )

    end_timer(fe_timer, "Feature Engineering")

    # ----------------------------------------------------
    # 5. Run EDA
    # ----------------------------------------------------
    eda_timer = start_timer()
    run_eda(df_final)
    end_timer(eda_timer, "EDA")

    # ----------------------------------------------------
    # 6. Train Model
    # IMPORTANT: Change target column based on your dataset!
    # ----------------------------------------------------
    TARGET = "num"  # <-- CHANGE THIS TO YOUR DATASET'S TARGET COLUMN NAME

    train_timer = start_timer()
    model, X_test, y_test = train_decision_tree(df_final, TARGET)
    end_timer(train_timer, "Training Decision Tree Model")

    if model is None:
        print("[FATAL] Model training failed. Pipeline cannot continue.")
        log_message("[FATAL] Model training failed. Aborting pipeline.")
        return

    # ----------------------------------------------------
    # 7. Evaluate Model
    # ----------------------------------------------------
    eval_timer = start_timer()
    evaluate_model(model, X_test, y_test)
    end_timer(eval_timer, "Model Evaluation")

    # ----------------------------------------------------
    # 8. Visualize Decision Tree
    # ----------------------------------------------------
    viz_timer = start_timer()

    feature_names = list(df_final.drop(columns=[TARGET]).columns)
    visualize_tree(model, feature_names)

    end_timer(viz_timer, "Decision Tree Visualization")

    # ----------------------------------------------------
    # 9. Completion Message
    # ----------------------------------------------------
    print("\n====================================================")
    print("           PIPELINE COMPLETED SUCCESSFULLY!")
    print("====================================================")
    log_message("[INFO] Pipeline completed successfully.")


# ============================================================
# RUN PIPELINE
# ============================================================

if __name__ == "__main__":
    run_pipeline()
