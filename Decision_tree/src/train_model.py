"""
train_model.py
----------------
This file trains the Decision Tree model.

What this script does:
✔ Splits the dataset into training and test sets
✔ Trains a Decision Tree Classifier
✔ Saves the trained model
✔ Returns model, X_test, y_test
"""

import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(df, target_col):
    """
    Train a Decision Tree model on the dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset AFTER preprocessing + feature engineering

    target_col : str
        The name of the target column (e.g., "num")

    Returns:
    --------
    model : trained DecisionTreeClassifier
    X_test : DataFrame
    y_test : Series
    """

    print("\n[INFO] Starting model training...")

    # ---------------------------------------------------------
    # 1. Validate target column
    # ---------------------------------------------------------
    if target_col not in df.columns:
        print(f"[ERROR] Target column '{target_col}' not found in dataset.")
        return None, None, None

    print("[INFO] Splitting dataset into X (features) and y (target)...")

    # Features = all columns except target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ---------------------------------------------------------
    # 2. Train-Test Split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[INFO] Training Decision Tree Classifier...")

    # ---------------------------------------------------------
    # 3. Create and train model
    # ---------------------------------------------------------
    model = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("[INFO] Model training completed successfully.")

    # ---------------------------------------------------------
    # 4. Save trained model
    # ---------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    model_path = "models/decision_tree_model.pkl"
    joblib.dump(model, model_path)

    print(f"[INFO] Model saved to: {model_path}")

    return model, X_test, y_test


# ============================================================
# TEST: Run this file independently
# ============================================================
if __name__ == "__main__":
    print("[TEST] Running train_model.py directly...")

    from load_data import load_raw_dataset
    from preprocess import preprocess_data
    from feature_engineering import feature_engineering

    df_raw = load_raw_dataset()
    df_clean = preprocess_data(df_raw)

    df_final = feature_engineering(
        df_clean,
        label_encode_cols=["sex", "fbs", "exang"],
        one_hot_cols=["cp", "restecg", "slope", "thal"],
        scale_cols=["age", "trestbps", "chol", "thalach", "oldpeak"]
    )

    model, X_test, y_test = train_decision_tree(df_final, "num")

    if model:
        print("[TEST] Model trained successfully.")
