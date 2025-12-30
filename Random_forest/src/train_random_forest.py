"""
train_random_forest.py
----------------------
This file trains a Random Forest Classifier on the Glass dataset.

Tasks performed:
1. Split dataset into train/test
2. Train a Random Forest model
3. Evaluate the model (accuracy, precision, recall, F1-score)
4. Save the trained model into: models/random_forest_model.pkl

Notes:
- Random Forest performs well even with noisy or unscaled data,
  but in our pipeline, we use the PREPROCESSED dataset.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import pickle


# -------------------------------------------------------------------------
# Helper: ensure models/ folder exists
# -------------------------------------------------------------------------
def ensure_models_folder(path="models/"):
    """
    Auto-creates the 'models' directory if missing.

    Why?
    ----
    - When saving trained models (.pkl files), the folder must exist.
    - Prevents FileNotFoundError.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[AUTO] Created folder: {path}")


# -------------------------------------------------------------------------
# Main Function: Train Random Forest
# -------------------------------------------------------------------------
def train_random_forest(df):
    """
    Trains a Random Forest classifier on the given dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
         Preprocessed dataset containing features + target

    Returns:
    --------
    model : trained RandomForestClassifier
    metrics : dict containing accuracy, precision, recall, f1-score

    Steps explained:
    ----------------
    1. Split dataset into X (features) and y (target)
    2. Train-test split → 80% train, 20% test
    3. Train RandomForestClassifier
    4. Predict on test data
    5. Compute evaluation metrics
    6. Save model to folder
    """

    print("\n[INFO] Starting Random Forest training...")

    # -----------------------------------------------------------------
    # 1. Split X and y
    # -----------------------------------------------------------------
    print("[INFO] Splitting dataset into features and target...")

    X = df.iloc[:, :-1]   # All columns except last one
    y = df.iloc[:, -1]    # Last column is target

    # -----------------------------------------------------------------
    # 2. Train-test split
    # -----------------------------------------------------------------
    print("[INFO] Performing train-test split (80% train, 20% test)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # -----------------------------------------------------------------
    # 3. Define and train the Random Forest model
    # -----------------------------------------------------------------
    print("[INFO] Training Random Forest Classifier...")

    model = RandomForestClassifier(
        n_estimators=150,       # Number of trees
        max_depth=None,         # Let trees grow fully
        random_state=42,
        n_jobs=-1               # Use all CPU cores
    )

    model.fit(X_train, y_train)

    # -----------------------------------------------------------------
    # 4. Predict on test data
    # -----------------------------------------------------------------
    print("[INFO] Making predictions...")

    y_pred = model.predict(X_test)

    # -----------------------------------------------------------------
    # 5. Evaluate model performance
    # -----------------------------------------------------------------
    print("[INFO] Evaluating model performance...")

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    print("\n===== RANDOM FOREST PERFORMANCE =====")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1-score   : {metrics['f1_score']:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # -----------------------------------------------------------------
    # 6. Save model to disk
    # -----------------------------------------------------------------
    ensure_models_folder()

    model_path = "models/random_forest_model.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print(f"[INFO] Random Forest model saved to: {model_path}")

    return model, metrics


# -------------------------------------------------------------------------
# TEST BLOCK — run this using:
# python src/train_random_forest.py
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running train_random_forest.py directly...")

    try:
        df_test = pd.read_csv("data/processed/cleaned_glass.csv")
        train_random_forest(df_test)
        print("\n[TEST] train_random_forest.py is working correctly ✔️")
    except Exception as e:
        print(f"[TEST ERROR] {e}")
