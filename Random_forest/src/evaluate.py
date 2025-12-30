"""
evaluate.py
------------
This file provides reusable functions to evaluate ML models.

What it does:
-------------
1. Accepts any trained classifier + test data
2. Computes evaluation metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
3. Generates a classification report
4. Saves evaluation results to: reports/model_performance.txt

Why this file exists:
---------------------
Instead of duplicating evaluation code in every model script,
we write one clean reusable evaluation module.
"""

import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# -------------------------------------------------------------------------
# Helper — ensure reports folder exists
# -------------------------------------------------------------------------
def ensure_reports_folder(path="reports/"):
    """
    Automatically creates the 'reports' folder if missing.

    Why?
    ----
    Without this, saving the evaluation report will fail with a
    FileNotFoundError. This ensures a smooth experience.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[AUTO] Created folder: {path}")


# -------------------------------------------------------------------------
# Main evaluation function
# -------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained model using standard ML metrics.

    Parameters:
    -----------
    model : trained ML classifier
    X_test : pandas DataFrame or numpy array
    y_test : pandas Series or array
    model_name : str (used for printing and report title)

    Returns:
    --------
    metrics : dict
        Contains accuracy, precision, recall, F1-score

    Explanation:
    ------------
    - Model predicts y_pred
    - All evaluation metrics are computed
    - Classification report is printed
    - Results are returned + saved to reports folder
    """

    print(f"\n[INFO] Evaluating {model_name}...")

    # --------------------------------------------------------------
    # 1. Predict on test data
    # --------------------------------------------------------------
    y_pred = model.predict(X_test)

    # --------------------------------------------------------------
    # 2. Compute evaluation metrics
    # --------------------------------------------------------------
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    # Display metrics in terminal
    print("\n========== MODEL PERFORMANCE ==========")
    print(f"Model Name : {model_name}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1-Score   : {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --------------------------------------------------------------
    # 3. Save evaluation report
    # --------------------------------------------------------------
    ensure_reports_folder()

    report_path = "reports/model_performance.txt"

    with open(report_path, "w") as file:
        file.write("=============== MODEL PERFORMANCE REPORT ===============\n")
        file.write(f"Model Name: {model_name}\n\n")
        file.write(f"Accuracy  : {metrics['accuracy']:.4f}\n")
        file.write(f"Precision : {metrics['precision']:.4f}\n")
        file.write(f"Recall    : {metrics['recall']:.4f}\n")
        file.write(f"F1 Score  : {metrics['f1_score']:.4f}\n\n")
        file.write("Classification Report:\n")
        file.write(classification_report(y_test, y_pred))
        file.write("========================================================\n")

    print(f"[INFO] Evaluation report saved to: {report_path}")

    return metrics


# -------------------------------------------------------------------------
# TEST BLOCK (run standalone):
# python src/evaluate.py
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running evaluate.py directly...")

    try:
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd

        # Load cleaned data and prepare a mini test (for testing this file)
        df_test = pd.read_csv("data/processed/cleaned_glass.csv")

        X = df_test.iloc[:, :-1]
        y = df_test.iloc[:, -1]

        model = RandomForestClassifier().fit(X, y)

        evaluate_model(model, X, y, "TEST Random Forest")

        print("\n[TEST] evaluate.py is working correctly ✔️")

    except Exception as e:
        print(f"[TEST ERROR] {e}")
