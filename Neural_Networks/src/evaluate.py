"""
evaluate.py
-----------
This file evaluates the performance of:

1. The BASELINE model   -> models/baseline_model.h5
2. The TUNED (best) model -> models/best_model.h5   (from hyperparameter tuning)

We calculate, for each model:
- Accuracy
- Precision
- Recall
- F1-score

And we compare:
- Baseline vs Tuned model performance

What this script does:
1. Load the processed dataset.
2. Split it into train and test sets (same way as train.py).
3. Load each saved model.
4. Get predictions on the test set.
5. Compute evaluation metrics.
6. Save metrics to JSON files and print them nicely.
"""

import os
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import tensorflow as tf

# Reuse functions from train.py to avoid rewriting logic
from train import (
    load_processed_data,       # loads processed CSV
    split_features_and_target, # splits into X and y (as integer labels)
    train_test_split_data      # creates train-test split
)

# Import config values (paths, etc.)
from config import (
    MODELS_DIR,               # folder where .h5 models are stored
    BASELINE_METRICS_PATH,    # JSON file for baseline metrics
    TUNED_METRICS_PATH,       # JSON file for tuned model metrics
    create_directories        # ensure folders exist
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Compute accuracy, precision, recall, and F1-score.

    Args:
        y_true: true class labels (integers)
        y_pred: predicted class labels (integers)

    Returns:
        A dictionary with overall (weighted) metrics and per-class metrics.
    """

    # Overall accuracy: fraction of correct predictions
    accuracy = accuracy_score(y_true, y_pred)

    # precision_recall_fscore_support with average="weighted":
    # - precision: how many predicted positives are actually correct
    # - recall   : how many actual positives we detected
    # - f1       : harmonic mean of precision and recall
    # "weighted" means it takes into account class imbalance.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Detailed per-class metrics (no averaging)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

    # Confusion matrix: rows = true class, columns = predicted class
    cm = confusion_matrix(y_true, y_pred)

    # Build the result dictionary
    metrics = {
        "overall": {
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
        },
        "per_class": {
            "precision": precision_per_class.tolist(),
            "recall": recall_per_class.tolist(),
            "f1": f1_per_class.tolist(),
            "support": support_per_class.tolist(),
        },
        "confusion_matrix": cm.tolist()
    }

    return metrics


def evaluate_model_on_test_set(
    model_path: str,
    X_test: np.ndarray,
    y_test_int: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """
    Load a Keras model from disk and evaluate it on the test set.

    Steps:
    1. Check if the model file exists.
    2. Load the model.
    3. Use model.predict() to get probability outputs.
    4. Convert probabilities to predicted class indices using argmax.
    5. Compute classification metrics.
    6. Print metrics and classification report.

    Args:
        model_path: path to the .h5 model file
        X_test: test feature data
        y_test_int: true test labels (integers)
        model_name: a friendly name to print (e.g., "Baseline", "Tuned")

    Returns:
        metrics_dict: dictionary with computed metrics (overall + per-class)
    """

    if not os.path.exists(model_path):
        # If the model doesn't exist, we cannot evaluate it.
        print(f"[WARNING] {model_name} model not found at: {model_path}")
        return {}

    print(f"\n[INFO] Loading {model_name} model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Use model to predict class probabilities for each sample in X_test.
    # Output shape: (num_samples, num_classes)
    print(f"[INFO] Generating predictions for {model_name}...")
    y_pred_proba = model.predict(X_test, verbose=0)

    # Convert probability vectors to predicted class indices:
    # Example: [0.1, 0.7, 0.2] -> class 1 (index of max value)
    y_pred_int = np.argmax(y_pred_proba, axis=1)

    # Compute metrics
    metrics_dict = compute_classification_metrics(y_test_int, y_pred_int)

    # Print important metrics to console
    print(f"\n========== {model_name} MODEL EVALUATION ==========")
    print(f"Accuracy (weighted): {metrics_dict['overall']['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics_dict['overall']['precision_weighted']:.4f}")
    print(f"Recall (weighted): {metrics_dict['overall']['recall_weighted']:.4f}")
    print(f"F1-score (weighted): {metrics_dict['overall']['f1_weighted']:.4f}")

    # Show classification report (per-class metrics, precision/recall/f1)
    print("\nClassification report (per class indices):")
    print(classification_report(y_test_int, y_pred_int, zero_division=0))

    # Show confusion matrix as raw numbers
    print("Confusion matrix (rows=true, cols=pred):")
    print(np.array(metrics_dict["confusion_matrix"]))
    print("====================================================\n")

    return metrics_dict


def merge_and_save_metrics(
    base_metrics_path: str,
    new_metrics: Dict[str, Any]
) -> None:
    """
    Merge existing metrics JSON file (if it exists) with new metrics,
    then save back to the same path.

    Why:
    - train.py already saved test_loss and test_accuracy.
    - Here we add precision, recall, F1, confusion matrix, etc.
    - We want all of them together in one JSON file.

    Args:
        base_metrics_path: path to metrics JSON (baseline or tuned)
        new_metrics: dictionary of new metrics to add/update
    """
    # Start with existing metrics if file exists
    if os.path.exists(base_metrics_path):
        with open(base_metrics_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
    else:
        base_data = {}

    # Update existing dictionary with new metrics (override or add)
    base_data.update(new_metrics)

    # Ensure directory exists
    os.makedirs(os.path.dirname(base_metrics_path), exist_ok=True)

    # Save the merged dictionary back to JSON
    with open(base_metrics_path, "w", encoding="utf-8") as f:
        json.dump(base_data, f, indent=4)

    print(f"[INFO] Saved updated metrics to: {base_metrics_path}")


def evaluate_baseline_and_tuned_models() -> None:
    """
    Full evaluation pipeline:

    1. Load processed data.
    2. Split into X and y (integer labels).
    3. Do train-test split (same as in training).
    4. Evaluate:
        - baseline_model.h5
        - best_model.h5 (tuned)
    5. Save metrics for both models as JSON.

    This allows you to compare:
    - How much the tuned model improved over the baseline.
    """

    # --------------------------------------------------------
    # 1. Load processed dataset
    # --------------------------------------------------------
    df_processed = load_processed_data()

    # --------------------------------------------------------
    # 2. Split into features (X) and labels (y as ints)
    # --------------------------------------------------------
    # We reuse the same function as in train.py so that
    # label encoding (class indices) is consistent.
    X, y_int = split_features_and_target(df_processed)

    # --------------------------------------------------------
    # 3. Create train-test split (same random_state as in training)
    # --------------------------------------------------------
    X_train, X_test, y_train_int, y_test_int = train_test_split_data(X, y_int)

    # NOTE: We don't need X_train or y_train_int here.
    # We only need X_test and y_test_int for evaluation.
    # But we call train_test_split_data to ensure we use
    # EXACTLY the same splitting logic as in train.py,
    # so that results are comparable.

    # --------------------------------------------------------
    # 4. Evaluate the BASELINE model
    # --------------------------------------------------------
    baseline_model_path = os.path.join(MODELS_DIR, "baseline_model.h5")
    baseline_metrics = evaluate_model_on_test_set(
        model_path=baseline_model_path,
        X_test=X_test,
        y_test_int=y_test_int,
        model_name="Baseline"
    )

    # If metrics dict not empty, merge & save with base baseline_metrics.json
    if baseline_metrics:
        merge_and_save_metrics(BASELINE_METRICS_PATH, {
            "classification_metrics": baseline_metrics
        })

    # --------------------------------------------------------
    # 5. Evaluate the TUNED (best) model
    # --------------------------------------------------------
    best_model_path = os.path.join(MODELS_DIR, "best_model.h5")
    tuned_metrics = evaluate_model_on_test_set(
        model_path=best_model_path,
        X_test=X_test,
        y_test_int=y_test_int,
        model_name="Tuned (Best)"
    )

    if tuned_metrics:
        merge_and_save_metrics(TUNED_METRICS_PATH, {
            "classification_metrics": tuned_metrics
        })

    # --------------------------------------------------------
    # 6. Print quick comparison if both models were evaluated
    # --------------------------------------------------------
    if baseline_metrics and tuned_metrics:
        base_acc = baseline_metrics["overall"]["accuracy"]
        tuned_acc = tuned_metrics["overall"]["accuracy"]

        print("============== BASELINE vs TUNED COMPARISON ==============")
        print(f"Baseline Accuracy: {base_acc:.4f}")
        print(f"Tuned    Accuracy: {tuned_acc:.4f}")
        print(f"Accuracy Improvement: {tuned_acc - base_acc:.4f}")
        print("===========================================================")


# ----------------------------------------------------------------------
# Entry point: run evaluation from terminal.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    If you run:

        python evaluate.py

    This will:
    1. Create necessary directories.
    2. Evaluate both:
        - baseline_model.h5
        - best_model.h5 (if available)
    3. Save metrics into:
        - output/metrics/baseline_metrics.json
        - output/metrics/tuned_metrics.json
    """

    # Ensure folder structure is ready
    create_directories()

    try:
        evaluate_baseline_and_tuned_models()
    except FileNotFoundError as e:
        # Most common cause: processed CSV missing or models not trained yet
        print(e)
