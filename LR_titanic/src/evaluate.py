"""
evaluate.py
-----------
This file is responsible for:

1. Loading the cleaned dataset (clean_train.csv)
2. Loading the trained model from /models/logistic_model.pkl
3. Evaluating the model performance on the whole dataset
4. Saving evaluation reports to /output/reports/
5. Saving plots (ROC Curve, Confusion Matrix) to /output/plots/

This is the THIRD major step in the ML pipeline.
"""

import os
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt


# ----------------------------------------------------
# Load data and model
# ----------------------------------------------------
def load_clean_data(path):
    """
    Loads cleaned Titanic data.
    """
    return pd.read_csv(path)


def load_model(path):
    """
    Loads the trained Logistic Regression model.
    IMPORTANT: Using joblib.load() because model was saved by joblib.dump()
    """
    return joblib.load(path)


# ----------------------------------------------------
# Save plots
# ----------------------------------------------------
def save_confusion_matrix(cm, output_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(output_path)
    plt.close()


def save_roc_curve(fpr, tpr, roc_auc, output_path):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


# ----------------------------------------------------
# Evaluate all metrics
# ----------------------------------------------------
def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    cm = confusion_matrix(y, y_pred)

    # ROC data
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    return accuracy, precision, recall, f1, cm, fpr, tpr, roc_auc


# ----------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    CLEAN_DATA_PATH = os.path.join(BASE_DIR, "output", "clean_train.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_model.pkl")

    REPORTS_DIR = os.path.join(BASE_DIR, "output", "reports")
    PLOTS_DIR = os.path.join(BASE_DIR, "output", "plots")

    # Make folders if missing (automatic)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("[INFO] Loading cleaned dataset...")
    df = load_clean_data(CLEAN_DATA_PATH)

    print("[INFO] Loading trained model...")
    model = load_model(MODEL_PATH)

    # Prepare data
    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    # Evaluate
    print("[INFO] Evaluating model...")
    accuracy, precision, recall, f1, cm, fpr, tpr, roc_auc = evaluate_model(model, X, y)

    # Save confusion matrix plot
    save_confusion_matrix(cm, os.path.join(PLOTS_DIR, "confusion_matrix.png"))

    # Save ROC curve
    save_roc_curve(fpr, tpr, roc_auc, os.path.join(PLOTS_DIR, "roc_curve.png"))

    # Save text report
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("=== MODEL EVALUATION REPORT ===\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write(f"ROC AUC  : {roc_auc:.4f}\n")

    print("[DONE] Evaluation complete!")
    print(f"Saved evaluation report to: {report_path}")
    print(f"Saved plots to: {PLOTS_DIR}")
