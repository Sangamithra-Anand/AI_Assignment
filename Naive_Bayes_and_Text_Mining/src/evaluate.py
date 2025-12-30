"""
evaluate.py
-----------------------------------------
This file evaluates the Naive Bayes model trained in train_model.py.

Steps performed:
1. Load trained model + TF-IDF vectorizer.
2. Load cleaned dataset and reapply TF-IDF.
3. Compute evaluation metrics:
        - Accuracy
        - Precision
        - Recall
        - F1-score
4. Generate and save confusion matrix plot.
5. Save evaluation report in /reports/

Everything is explained inside the code using comments.
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from feature_engineering import load_cleaned_dataset, apply_tfidf
from train_model import ensure_models_folder


# ----------------------------------------------------------------------
# Helper: ensure reports folder exists
# ----------------------------------------------------------------------
def ensure_reports_folder(path="reports/"):
    """
    Creates 'reports/' folder if it does not exist.
    Prevents file-saving errors when generating evaluation outputs.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[AUTO] Created folder: {path}")


# ----------------------------------------------------------------------
# Load trained model + TF-IDF vectorizer
# ----------------------------------------------------------------------
def load_model_and_vectorizer(
    model_path="models/naive_bayes_model.pkl",
    vectorizer_path="models/tfidf_vectorizer.pkl"
):
    """
    Loads the trained Naive Bayes model and TF-IDF vectorizer.

    Returns:
        model, vectorizer
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("[HINT] Run train_model.py first.")
        return None, None

    if not os.path.exists(vectorizer_path):
        print(f"[ERROR] Vectorizer file not found: {vectorizer_path}")
        print("[HINT] Run train_model.py first.")
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    print("[INFO] Model and vectorizer loaded successfully.")
    return model, vectorizer


# ----------------------------------------------------------------------
# Main evaluation function
# ----------------------------------------------------------------------
def evaluate_model():
    """
    Evaluates the trained Naive Bayes model using:
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Confusion Matrix (saved as image)

    Saves report to /reports/
    """

    print("[INFO] Loading cleaned dataset...")
    df = load_cleaned_dataset()
    if df is None:
        return

    print("[INFO] Loading model + vectorizer...")
    model, vectorizer = load_model_and_vectorizer()
    if model is None:
        return

    print("[INFO] Applying TF-IDF using loaded vectorizer...")
    X = vectorizer.transform(df["Cleaned_Text"])
    y = df["Labels"]

    # Split TF-IDF results into train/test in SAME WAY as before
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("[INFO] Making predictions...")
    y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # Compute evaluation metrics
    # ---------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    print("[INFO] Evaluation Metrics:")
    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1-score : {f1:.4f}")

    # ---------------------------------------------------------
    # Generate confusion matrix
    # ---------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    ensure_reports_folder()
    cm_path = "reports/confusion_matrix.png"

    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved confusion matrix at: {cm_path}")

    # ---------------------------------------------------------
    # Save evaluation report
    # ---------------------------------------------------------
    report_path = "reports/classification_report.txt"
    with open(report_path, "w") as f:
        f.write("MODEL EVALUATION REPORT\n")
        f.write("-----------------------\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")

    print(f"[INFO] Saved evaluation report: {report_path}")


# ----------------------------------------------------------------------
# Self-Test Block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running evaluate.py directly...")
    evaluate_model()
    print("[TEST] Evaluation completed successfully.")
