"""
regularization.py
------------------
This script applies Lasso and Ridge regression to the Toyota Corolla dataset.

Why Regularization?
-------------------
Multiple Linear Regression often suffers from:
- Multicollinearity (high correlation between features)
- Overfitting

Lasso (L1):
    Shrinks coefficients and can force some to become zero.
    → Helps in feature selection.

Ridge (L2):
    Shrinks coefficients but does NOT force them to zero.
    → Helps when features are correlated.

This script:
1. Loads cleaned dataset
2. Scales features (required for regularization)
3. Applies LassoCV and RidgeCV for optimal model tuning
4. Saves models into models/
5. Saves coefficient summary into output/
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import ensure_directory


# =====================================================================================
# Function: train_regularization_models
# =====================================================================================
def train_regularization_models(cleaned_df, models_dir="models", output_dir="output"):
    """
    Trains Lasso and Ridge regression models and saves results.

    Parameters:
    cleaned_df (DataFrame): Preprocessed Toyota Corolla dataset.
    models_dir (str): Folder where trained models will be stored.
    output_dir (str): Folder to save output summaries.

    Returns:
    dict containing trained Lasso and Ridge models.
    """

    print("\n[INFO] Starting Lasso and Ridge Regression Training...")

    # ------------------------------------------------------------------------------
    # 1. Split Dataset into Features (X) and Target (y)
    # ------------------------------------------------------------------------------
    X = cleaned_df.drop("Price", axis=1)
    y = cleaned_df["Price"]

    print("[INFO] Performing train-test split for regularization models...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ------------------------------------------------------------------------------
    # 2. Scale Features (VERY IMPORTANT for Lasso & Ridge)
    # ------------------------------------------------------------------------------
    """
    Lasso & Ridge are distance-based methods.
    If one feature has a larger scale (e.g., KM = 100,000 vs. Doors = 3),
    it will dominate the loss function.

    StandardScaler → transforms data to mean = 0, std = 1.
    """

    print("[INFO] Scaling features before regularization...")
    scaler = StandardScaler()

    # Fit only on training data to avoid data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for future predictions
    ensure_directory(models_dir)
    scaler_path = os.path.join(models_dir, "regularization_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Saved scaler used for regularization models at: {scaler_path}")

    # ------------------------------------------------------------------------------
    # 3. Train Lasso Regression with Cross Validation
    # ------------------------------------------------------------------------------
    """
    LassoCV automatically finds the best alpha (penalty strength)
    by testing multiple values using cross validation.
    """

    print("[INFO] Training LassoCV model...")
    lasso_model = LassoCV(cv=5, random_state=42)
    lasso_model.fit(X_train_scaled, y_train)

    # Save Lasso model
    lasso_path = os.path.join(models_dir, "lasso_model.pkl")
    joblib.dump(lasso_model, lasso_path)
    print(f"[INFO] Lasso model saved at: {lasso_path}")

    # ------------------------------------------------------------------------------
    # 4. Train Ridge Regression with Cross Validation
    # ------------------------------------------------------------------------------
    """
    RidgeCV tests multiple alpha values and picks the best one automatically.
    """

    print("[INFO] Training RidgeCV model...")
    alphas = [0.1, 1, 5, 10, 50, 100]

    ridge_model = RidgeCV(alphas=alphas, cv=5)
    ridge_model.fit(X_train_scaled, y_train)

    # Save Ridge model
    ridge_path = os.path.join(models_dir, "ridge_model.pkl")
    joblib.dump(ridge_model, ridge_path)
    print(f"[INFO] Ridge model saved at: {ridge_path}")

    # ------------------------------------------------------------------------------
    # 5. Save Summary of Regularization Results
    # ------------------------------------------------------------------------------
    ensure_directory(output_dir)
    summary_path = os.path.join(output_dir, "regularization_summary.txt")

    print(f"[INFO] Saving regularization summary at: {summary_path}")

    with open(summary_path, "w") as f:
        f.write("=== LASSO & RIDGE REGRESSION SUMMARY ===\n\n")

        # Lasso Summary
        f.write("LASSO REGRESSION:\n")
        f.write(f"Best Alpha: {lasso_model.alpha_}\n")
        f.write("Coefficients:\n")
        f.write(str(dict(zip(X.columns, lasso_model.coef_))))
        f.write("\n\n")

        # Ridge Summary
        f.write("RIDGE REGRESSION:\n")
        f.write(f"Best Alpha: {ridge_model.alpha_}\n")
        f.write("Coefficients:\n")
        f.write(str(dict(zip(X.columns, ridge_model.coef_))))
        f.write("\n\n")

    print("\n[INFO] Lasso & Ridge Training Completed Successfully!")

    return {
        "lasso_model": lasso_model,
        "ridge_model": ridge_model
    }


# =====================================================================================
# Direct Execution: Allows running this file alone
# =====================================================================================
if __name__ == "__main__":
    print("[TEST] Running regularization.py directly...")

    cleaned_path = "data/processed/cleaned_data.csv"

    if os.path.exists(cleaned_path):
        df_clean = pd.read_csv(cleaned_path)
        train_regularization_models(df_clean)
    else:
        print("[ERROR] Cleaned dataset not found. Run preprocess.py first.")

