"""
EVALUATION SCRIPT
------------------
This file:

1. Calculates metric scores
2. Saves LightGBM report
3. Saves XGBoost report
4. Creates comparison_report.txt
"""

import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ------------------------------------------------------------
# CALCULATE METRICS
# ------------------------------------------------------------
def calculate_metrics(model, X_val, y_val):

    preds = model.predict(X_val)

    return {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
        "f1_score": f1_score(y_val, preds)
    }


# ------------------------------------------------------------
# SAVE A SINGLE REPORT
# ------------------------------------------------------------
def save_report(metrics, filename):

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    report_dir = os.path.join(BASE_DIR, "output", "reports")
    os.makedirs(report_dir, exist_ok=True)

    path = os.path.join(report_dir, filename)

    with open(path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"Saved: {path}")


# ------------------------------------------------------------
# EVALUATE BOTH MODELS
# ------------------------------------------------------------
def evaluate_models(lgb_model, xgb_model, X_val, y_val):

    print("\nEvaluating LightGBM...")
    lgb_metrics = calculate_metrics(lgb_model, X_val, y_val)
    save_report(lgb_metrics, "lightgbm_evaluation.txt")

    print("\nEvaluating XGBoost...")
    xgb_metrics = calculate_metrics(xgb_model, X_val, y_val)
    save_report(xgb_metrics, "xgboost_evaluation.txt")

    # Create combined comparison report
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    compare_path = os.path.join(BASE_DIR, "output", "reports", "comparison_report.txt")
    os.makedirs(os.path.dirname(compare_path), exist_ok=True)

    with open(compare_path, "w") as f:
        f.write("LIGHTGBM vs XGBOOST REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("[LightGBM]\n")
        for k, v in lgb_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\n[XGBoost]\n")
        for k, v in xgb_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"\nSaved: {compare_path}")
    print("\nEvaluation Completed Successfully!")


