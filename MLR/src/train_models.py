"""
train_models.py
----------------
This script trains multiple Linear Regression models for the Toyota Corolla dataset.

Models trained:
1. Model 1 — Basic Linear Regression with ALL features
2. Model 2 — Reduced Model (after VIF-based multicollinearity removal)
3. Model 3 — Standardized Model (Scaled features)

All trained models are saved inside the 'models/' directory.
Coefficient summaries are saved inside 'output/' directory.
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import ensure_directory, calculate_vif


# =====================================================================
# Helper Function: Train & Save Model
# =====================================================================
def train_and_save_model(X_train, y_train, model_name, save_dir="models"):
    """
    Trains a Linear Regression model and saves it to disk.

    Parameters:
        X_train: Training features
        y_train: Training target
        model_name (str): filename
        save_dir (str): directory to save model

    Returns:
        model: trained LinearRegression model
    """
    print(f"[INFO] Training {model_name} ...")

    model = LinearRegression()
    model.fit(X_train, y_train)

    ensure_directory(save_dir)
    path = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(model, path)

    print(f"[INFO] Saved {model_name} at {path}\n")
    return model


# =====================================================================
# Main Training Function
# =====================================================================
def train_models(df, output_dir="output"):
    """
    Trains multiple regression models and handles VIF for reduced model.

    Parameters:
        df (DataFrame): Cleaned dataset

    Returns:
        dict: trained models
    """

    print("\n[INFO] Starting Model Training...")

    # ----------------------------------------------------------
    # 1. Split Features and Target
    # ----------------------------------------------------------
    print("[INFO] Splitting features and target column...")
    X = df.drop("Price", axis=1)
    y = df["Price"]

    print("[INFO] Performing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ----------------------------------------------------------
    # MODEL 1 — BASIC LINEAR REGRESSION
    # ----------------------------------------------------------
    model_1 = train_and_save_model(X_train, y_train, "model_1_basic")

    # ----------------------------------------------------------
    # MODEL 2 — REDUCED MODEL USING VIF
    # ----------------------------------------------------------
    print("[INFO] Calculating VIF and removing multicollinear features...")

    # Select only numeric columns
    X_vif = X_train.select_dtypes(include=["int64", "float64", "uint8", "bool"]).copy()

    # Convert boolean → int
    for col in X_vif.select_dtypes(include=["bool"]).columns:
        X_vif[col] = X_vif[col].astype(int)

    # Convert all columns to float (statsmodels requirement)
    X_vif = X_vif.astype(float)

    # Calculate VIF
    vif_df = calculate_vif(X_vif)

    # Identify features to drop (VIF > 10)
    high_vif_features = vif_df[vif_df["VIF"] > 10]["feature"].tolist()

    print(f"[INFO] High VIF features to remove: {high_vif_features}")

    # Remove high VIF features
    X_train_reduced = X_vif.drop(columns=high_vif_features, errors="ignore")

    # Train Reduced Model
    model_2 = train_and_save_model(
        X_train_reduced, y_train, "model_2_reduced"
    )

    # ----------------------------------------------------------
    # MODEL 3 — STANDARDIZED MODEL (Scaled Features)
    # ----------------------------------------------------------
    print("[INFO] Training scaled model (StandardScaler)...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    ensure_directory("models")
    joblib.dump(scaler, "models/scaler.pkl")

    model_3 = train_and_save_model(
        X_train_scaled, y_train, "model_3_scaled"
    )

    # ----------------------------------------------------------
    # Save Coefficient Summary
    # ----------------------------------------------------------
    ensure_directory(output_dir)

    coef_summary_path = os.path.join(output_dir, "coefficient_summary.txt")
    print(f"[INFO] Saving coefficient summary at: {coef_summary_path}")

    with open(coef_summary_path, "w") as f:
        f.write("=== Model 1 Coefficients ===\n")
        f.write(str(dict(zip(X.columns, model_1.coef_))) + "\n\n")

        f.write("=== High VIF Features Removed ===\n")
        f.write(str(high_vif_features) + "\n\n")

        f.write("=== Model 2 Coefficients (Reduced Model) ===\n")
        f.write(str(dict(zip(X_train_reduced.columns, model_2.coef_))) + "\n\n")

        f.write("=== Model 3 Coefficients (Scaled Model) ===\n")
        f.write(str(model_3.coef_) + "\n")

    print("\n[INFO] Training Completed Successfully!\n")

    return {
        "model_1": model_1,
        "model_2": model_2,
        "model_3": model_3,
    }


# =====================================================================
# Direct Run Test
# =====================================================================
if __name__ == "__main__":
    print("[TEST] Running train_models.py directly...")

    path = "data/processed/cleaned_data.csv"

    if os.path.exists(path):
        df_clean = pd.read_csv(path)
        train_models(df_clean)
    else:
        print("[ERROR] No cleaned_data.csv found. Run preprocess.py first.")
