"""
evaluate.py
------------
This file evaluates the trained Decision Tree model.

What this script does:
✔ Takes the trained model, X_test, y_test
✔ Calculates accuracy, precision, recall, F1-score
✔ Generates and saves confusion matrix plot
✔ Saves evaluation metrics into outputs/metrics.json
✔ Prints results to console

Each step includes explanations inside the code.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model with multiple metrics.

    Parameters:
    -----------
    model : DecisionTreeClassifier
        The trained model loaded from train_model.py

    X_test : pandas.DataFrame
        Testing features

    y_test : pandas.Series
        Testing target values

    Returns:
    --------
    metrics : dict
        Contains accuracy, precision, recall, f1-score
    """

    print("\n[INFO] Starting model evaluation...")

    #-----------------------------------------------------------
    # 1. Validation: Ensure model and test data are available
    #-----------------------------------------------------------
    if model is None or X_test is None or y_test is None:
        print("[ERROR] Evaluation failed — model or test data missing.")
        return None

    #-----------------------------------------------------------
    # 2. Make predictions using the trained model
    #-----------------------------------------------------------
    print("[INFO] Making predictions on test data...")
    y_pred = model.predict(X_test)

    #-----------------------------------------------------------
    # 3. Calculate evaluation metrics
    #-----------------------------------------------------------
    print("[INFO] Calculating performance metrics...")

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # Print metrics nicely
    print("\n===== MODEL PERFORMANCE =====")
    for key, value in metrics.items():
        print(f"{key.upper()}: {value:.4f}")

    #-----------------------------------------------------------
    # 4. Classification report (detailed performance)
    #-----------------------------------------------------------
    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred, zero_division=0))

    #-----------------------------------------------------------
    # 5. Save metrics to outputs/metrics.json
    #-----------------------------------------------------------
    outputs_path = "outputs"
    os.makedirs(outputs_path, exist_ok=True)

    metrics_file = os.path.join(outputs_path, "metrics.json")

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Metrics saved to: {metrics_file}")

    #-----------------------------------------------------------
    # 6. Create and save confusion matrix plot
    #-----------------------------------------------------------
    print("[INFO] Creating confusion matrix plot...")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = os.path.join(outputs_path, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Confusion matrix saved to: {cm_path}")

    print("[INFO] Evaluation completed successfully.\n")

    return metrics


# ============================================================
# TEST: Run evaluate.py directly for debugging
# ============================================================
if __name__ == "__main__":
    print("[TEST] Running evaluate.py directly...")

    from load_data import load_raw_dataset
    from preprocess import preprocess_data
    from train_model import train_decision_tree

    raw_df = load_raw_dataset()

    if raw_df is not None:
        clean_df = preprocess_data(raw_df)

        TARGET = "target"  # <-- CHANGE TO MATCH YOUR REAL TARGET COLUMN NAME

        model, X_test, y_test = train_decision_tree(clean_df, TARGET)

        if model:
            evaluate_model(model, X_test, y_test)
        else:
            print("[TEST ERROR] Model training failed, evaluation skipped.")
