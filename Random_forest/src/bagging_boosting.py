"""
bagging_boosting.py
-------------------
This file trains and evaluates two ensemble learning methods:

1. Bagging (Bootstrap Aggregating)
2. Boosting (AdaBoost Classifier)

For each model, we:
- Perform train/test split
- Train the ensemble model
- Make predictions
- Compute evaluation metrics (accuracy, precision, recall, F1-score)
- Save the trained model into /models/
- Save a comparison report into /reports/

Notes:
------
- Bagging reduces variance by training multiple independent models.
- Boosting reduces bias by training sequential models that correct errors.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# -------------------------------------------------------------------------
# Helper: Ensure folders exist
# -------------------------------------------------------------------------
def ensure_folders():
    """Auto-creates 'models/' and 'reports/' folders if missing."""
    folders = ["models", "reports"]
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)
            print(f"[AUTO] Created folder: {f}")


# -------------------------------------------------------------------------
# Helper: Evaluate model
# -------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates the model and prints + returns metrics.

    Returns:
    --------
    dict with accuracy, precision, recall, f1
    """

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    # Print results
    print(f"\n===== {model_name} PERFORMANCE =====")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1-score   : {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


# -------------------------------------------------------------------------
# Main Function: Train Bagging + Boosting
# -------------------------------------------------------------------------
def run_bagging_and_boosting(df):
    """
    Trains both Bagging and Boosting models and compares their performance.

    Steps:
    ------
    1. Split X and y
    2. Train BaggingClassifier
    3. Train AdaBoostClassifier
    4. Evaluate both models
    5. Save models into /models/
    6. Save comparison into /reports/
    """

    print("\n[INFO] Starting Bagging & Boosting pipeline...")

    ensure_folders()

    # --------------------------------------------------------
    # 1. Split data into X and y
    # --------------------------------------------------------
    print("[INFO] Preparing features and target...")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # --------------------------------------------------------
    # 2. Train Bagging Classifier
    # --------------------------------------------------------
    print("\n[INFO] Training Bagging Classifier...")

    bagging_model = BaggingClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
    )

    bagging_model.fit(X_train, y_train)

    bagging_metrics = evaluate_model(bagging_model, X_test, y_test, "Bagging Classifier")

    # Save model
    with open("models/bagging_model.pkl", "wb") as f:
        pickle.dump(bagging_model, f)
    print("[INFO] Bagging model saved to models/bagging_model.pkl")

    # --------------------------------------------------------
    # 3. Train Boosting Classifier (AdaBoost)
    # --------------------------------------------------------
    print("\n[INFO] Training AdaBoost Classifier...")

    boosting_model = AdaBoostClassifier(
        n_estimators=80,
        learning_rate=0.8,
        random_state=42,
    )

    boosting_model.fit(X_train, y_train)

    boosting_metrics = evaluate_model(boosting_model, X_test, y_test, "Boosting Classifier")

    # Save model
    with open("models/boosting_model.pkl", "wb") as f:
        pickle.dump(boosting_model, f)
    print("[INFO] Boosting model saved to models/boosting_model.pkl")

    # --------------------------------------------------------
    # 4. Save comparison report
    # --------------------------------------------------------
    comparison_path = "reports/comparison_results.txt"

    with open(comparison_path, "w") as file:
        file.write("================= BAGGING & BOOSTING COMPARISON =================\n\n")
        file.write("Bagging Classifier Metrics:\n")
        file.write(str(bagging_metrics) + "\n\n")
        file.write("Boosting Classifier Metrics:\n")
        file.write(str(boosting_metrics) + "\n\n")
        file.write("=================================================================\n")

    print(f"[INFO] Comparison report saved to: {comparison_path}")

    return bagging_metrics, boosting_metrics


# -------------------------------------------------------------------------
# TEST BLOCK — run using:
# python src/bagging_boosting.py
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running bagging_boosting.py directly...")

    try:
        df_test = pd.read_csv("data/processed/cleaned_glass.csv")
        run_bagging_and_boosting(df_test)
        print("\n[TEST] bagging_boosting.py is working correctly ✔️")
    except Exception as e:
        print(f"[TEST ERROR] {e}")
"""
bagging_boosting.py
-------------------
This file trains and evaluates two ensemble learning methods:

1. Bagging (Bootstrap Aggregating)
2. Boosting (AdaBoost Classifier)

For each model, we:
- Perform train/test split
- Train the ensemble model
- Make predictions
- Compute evaluation metrics (accuracy, precision, recall, F1-score)
- Save the trained model into /models/
- Save a comparison report into /reports/

Notes:
------
- Bagging reduces variance by training multiple independent models.
- Boosting reduces bias by training sequential models that correct errors.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# -------------------------------------------------------------------------
# Helper: Ensure folders exist
# -------------------------------------------------------------------------
def ensure_folders():
    """Auto-creates 'models/' and 'reports/' folders if missing."""
    folders = ["models", "reports"]
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)
            print(f"[AUTO] Created folder: {f}")


# -------------------------------------------------------------------------
# Helper: Evaluate model
# -------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates the model and prints + returns metrics.

    Returns:
    --------
    dict with accuracy, precision, recall, f1
    """

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    # Print results
    print(f"\n===== {model_name} PERFORMANCE =====")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1-score   : {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


# -------------------------------------------------------------------------
# Main Function: Train Bagging + Boosting
# -------------------------------------------------------------------------
def run_bagging_and_boosting(df):
    """
    Trains both Bagging and Boosting models and compares their performance.

    Steps:
    ------
    1. Split X and y
    2. Train BaggingClassifier
    3. Train AdaBoostClassifier
    4. Evaluate both models
    5. Save models into /models/
    6. Save comparison into /reports/
    """

    print("\n[INFO] Starting Bagging & Boosting pipeline...")

    ensure_folders()

    # --------------------------------------------------------
    # 1. Split data into X and y
    # --------------------------------------------------------
    print("[INFO] Preparing features and target...")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # --------------------------------------------------------
    # 2. Train Bagging Classifier
    # --------------------------------------------------------
    print("\n[INFO] Training Bagging Classifier...")

    bagging_model = BaggingClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
    )

    bagging_model.fit(X_train, y_train)

    bagging_metrics = evaluate_model(bagging_model, X_test, y_test, "Bagging Classifier")

    # Save model
    with open("models/bagging_model.pkl", "wb") as f:
        pickle.dump(bagging_model, f)
    print("[INFO] Bagging model saved to models/bagging_model.pkl")

    # --------------------------------------------------------
    # 3. Train Boosting Classifier (AdaBoost)
    # --------------------------------------------------------
    print("\n[INFO] Training AdaBoost Classifier...")

    boosting_model = AdaBoostClassifier(
        n_estimators=80,
        learning_rate=0.8,
        random_state=42,
    )

    boosting_model.fit(X_train, y_train)

    boosting_metrics = evaluate_model(boosting_model, X_test, y_test, "Boosting Classifier")

    # Save model
    with open("models/boosting_model.pkl", "wb") as f:
        pickle.dump(boosting_model, f)
    print("[INFO] Boosting model saved to models/boosting_model.pkl")

    # --------------------------------------------------------
    # 4. Save comparison report
    # --------------------------------------------------------
    comparison_path = "reports/comparison_results.txt"

    with open(comparison_path, "w") as file:
        file.write("================= BAGGING & BOOSTING COMPARISON =================\n\n")
        file.write("Bagging Classifier Metrics:\n")
        file.write(str(bagging_metrics) + "\n\n")
        file.write("Boosting Classifier Metrics:\n")
        file.write(str(boosting_metrics) + "\n\n")
        file.write("=================================================================\n")

    print(f"[INFO] Comparison report saved to: {comparison_path}")

    return bagging_metrics, boosting_metrics


# -------------------------------------------------------------------------
# TEST BLOCK — run using:
# python src/bagging_boosting.py
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running bagging_boosting.py directly...")

    try:
        df_test = pd.read_csv("data/processed/cleaned_glass.csv")
        run_bagging_and_boosting(df_test)
        print("\n[TEST] bagging_boosting.py is working correctly ✔️")
    except Exception as e:
        print(f"[TEST ERROR] {e}")
