"""
evaluate.py
------------
Evaluates all trained regression models:
1. Basic model
2. Reduced VIF-based model
3. Scaled model

Saves:
- MAE, MSE, RMSE, R²
- Evaluation summary
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils import ensure_directory, calculate_vif


# =====================================================================
# Helper: Evaluate a regression model
# =====================================================================
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    return mae, mse, rmse, r2


# =====================================================================
# Main evaluator
# =====================================================================
def evaluate_all_models(df, output_dir="output"):
    print("\n[INFO] Starting model evaluation...")

    # ----------------------------------------------------------
    # Train-test split (must match training order)
    # ----------------------------------------------------------
    print("[INFO] Performing train-test split for evaluation...")
    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ----------------------------------------------------------
    # Load trained models
    # ----------------------------------------------------------
    print("[INFO] Loading trained models...")

    model_1 = joblib.load("models/model_1_basic.pkl")
    model_2 = joblib.load("models/model_2_reduced.pkl")
    model_3 = joblib.load("models/model_3_scaled.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # ----------------------------------------------------------
    # PREPARE REDUCED TEST FEATURES (VIF < 10)
    # ----------------------------------------------------------
    print("[INFO] Preparing reduced test features (VIF < 10)...")

    # Select only numeric columns
    X_vif = X_train.select_dtypes(include=["int64", "float64", "uint8", "bool"]).copy()

    # Convert boolean → int
    for col in X_vif.select_dtypes(include=["bool"]).columns:
        X_vif[col] = X_vif[col].astype(int)

    # Convert everything to float
    X_vif = X_vif.astype(float)

    # Calculate VIF on training data
    vif_df = calculate_vif(X_vif)

    # Keep only low VIF features
    low_vif_features = vif_df[vif_df["VIF"] <= 10]["feature"].tolist()

    # Use the same reduced columns for test set
    X_test_reduced = X_test[low_vif_features]

    # ----------------------------------------------------------
    # Evaluate each model
    # ----------------------------------------------------------
    print("[INFO] Evaluating models...")

    # Model 1 — Basic
    m1 = evaluate_model(model_1, X_test, y_test)

    # Model 2 — Reduced
    m2 = evaluate_model(model_2, X_test_reduced, y_test)

    # Model 3 — Scaled
    X_test_scaled = scaler.transform(X_test)
    m3 = evaluate_model(model_3, X_test_scaled, y_test)

    # ----------------------------------------------------------
    # Save Evaluation Report
    # ----------------------------------------------------------
    ensure_directory(output_dir)

    eval_path = os.path.join(output_dir, "evaluation_results.txt")
    print(f"[INFO] Saving evaluation report at: {eval_path}")

    with open(eval_path, "w") as f:
        f.write("=== MODEL EVALUATION RESULTS ===\n\n")

        f.write("Model 1 — Basic Linear Regression:\n")
        f.write(f"MAE: {m1[0]:.4f}, MSE: {m1[1]:.4f}, RMSE: {m1[2]:.4f}, R2: {m1[3]:.4f}\n\n")

        f.write("Model 2 — Reduced VIF Model:\n")
        f.write(f"MAE: {m2[0]:.4f}, MSE: {m2[1]:.4f}, RMSE: {m2[2]:.4f}, R2: {m2[3]:.4f}\n\n")

        f.write("Model 3 — Scaled Model:\n")
        f.write(f"MAE: {m3[0]:.4f}, MSE: {m3[1]:.4f}, RMSE: {m3[2]:.4f}, R2: {m3[3]:.4f}\n\n")

    print("\n[INFO] Evaluation completed successfully!\n")

    return {
        "model_1": m1,
        "model_2": m2,
        "model_3": m3,
    }


# =====================================================================
# Direct Test
# =====================================================================
if __name__ == "__main__":
    print("[TEST] Running evaluate.py directly...")

    cleaned_path = "data/processed/cleaned_data.csv"

    if os.path.exists(cleaned_path):
        df_clean = pd.read_csv(cleaned_path)
        evaluate_all_models(df_clean)
    else:
        print("[ERROR] Cleaned dataset not found. Run preprocess.py first.")


