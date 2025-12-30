"""
evaluate.py
----------------------
This module evaluates the performance of the trained KNN classifier.

It generates:
    1. Accuracy
    2. Precision
    3. Recall
    4. F1-score
    5. Confusion Matrix (saved as an image)
    6. Classification Report (saved as a text file)

Everything is explained inside the code.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# FUNCTION: save_report()
# Purpose :
#     - Saves any report text inside output/reports/
# =========================================================
def save_report(filename, content):
    os.makedirs("output/reports", exist_ok=True)
    file_path = f"output/reports/{filename}"

    with open(file_path, "w") as f:
        f.write(content)

    print(f"[INFO] Report saved: {file_path}")


# =========================================================
# FUNCTION: plot_confusion_matrix()
# Purpose :
#     - Creates a confusion matrix plot and saves it.
#     - Displays how well KNN predicts each class.
# =========================================================
def plot_confusion_matrix(y_true, y_pred, labels):
    print("[INFO] Generating confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    os.makedirs("output/reports", exist_ok=True)
    plt.savefig("output/reports/confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("[INFO] Confusion matrix saved: output/reports/confusion_matrix.png")

    plt.close()


# =========================================================
# FUNCTION: evaluate_model()
# Purpose :
#     - Central evaluation pipeline.
#     - Calculates all metrics.
#     - Saves confusion matrix + text report.
#
# Parameters:
#     model : trained KNN model
#     X_test : testing features
#     y_test : true testing labels
#     label_names : list of class names (animal types)
# =========================================================
def evaluate_model(model, X_test, y_test, label_names=None):
    print("\n[INFO] Evaluating KNN model...")

    # -----------------------------
    # Step 1: Model predictions
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # Step 2: Metric calculations
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"[INFO] Accuracy:  {accuracy:.4f}")
    print(f"[INFO] Precision: {precision:.4f}")
    print(f"[INFO] Recall:    {recall:.4f}")
    print(f"[INFO] F1-score:  {f1:.4f}")

    # -----------------------------
    # Step 3: Create classification report
    # -----------------------------
    report_text = classification_report(y_test, y_pred, target_names=label_names)
    save_report("classification_report.txt", report_text)

    # -----------------------------
    # Step 4: Save metrics summary to JSON-like format
    # -----------------------------
    metrics_summary = (
        f"Accuracy: {accuracy}\n"
        f"Precision: {precision}\n"
        f"Recall: {recall}\n"
        f"F1 Score: {f1}\n"
    )

    save_report("metrics.txt", metrics_summary)

    # -----------------------------
    # Step 5: Confusion matrix plot
    # -----------------------------
    if label_names is None:
        label_names = sorted(list(set(y_test)))

    plot_confusion_matrix(y_test, y_pred, labels=label_names)

    print("[INFO] Model evaluation completed.")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


# =========================================================
# TEST BLOCK
# Runs only when executing:
#     python src/evaluate.py
# (Used for checking errors before full integration)
# =========================================================
if __name__ == "__main__":
    print("[TEST] Testing evaluate.py...")

    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split
        import pandas as pd

        df = pd.read_csv("data/Zoo.csv")

        X = df.drop(columns=["type"])
        y = df["type"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        evaluate_model(model, X_test, y_test)

        print("[TEST] Evaluation test completed.")

    except Exception as e:
        print("[TEST ERROR] Could not run evaluation test:", e)


