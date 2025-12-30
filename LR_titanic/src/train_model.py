"""
train_model.py
--------------
This file is responsible for:
1. Loading the cleaned dataset produced by preprocessing.py
2. Splitting the dataset into Train and Test sets
3. Training a Logistic Regression model
4. Evaluating the model performance
5. Saving the trained model to /models/logistic_model.pkl

This is the SECOND major step in the ML pipeline.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib   # used to save the trained model


def load_clean_data(path):
    """
    Loads the already cleaned dataset (clean_train.csv).
    Returns a pandas DataFrame.
    """
    return pd.read_csv(path)


def train_model(X, y):
    """
    Trains a Logistic Regression model on the given features X and labels y.
    Returns the trained model.
    """

    # Create Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    # Fit the model to the training data
    model.fit(X, y)

    return model


def save_model(model, output_path):
    """
    Saves the trained model to the given location using joblib.
    """
    joblib.dump(model, output_path)


if __name__ == "__main__":
    """
    This runs ONLY when executing:

        python train_model.py

    It will:
    - Load cleaned dataset
    - Prepare features and target
    - Train model
    - Evaluate accuracy, precision, recall, f1
    - Save model to /models/
    """

    # Build paths
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    CLEAN_DATA_PATH = os.path.join(BASE_DIR, "output", "clean_train.csv")
    MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "models", "logistic_model.pkl")

    print("[INFO] Loading cleaned dataset...")
    df = load_clean_data(CLEAN_DATA_PATH)

    # -----------------------------------------
    # Prepare features (X) and target (y)
    # -----------------------------------------

    # y = target column
    y = df["Survived"]

    # X = every column except Survived
    X = df.drop(columns=["Survived"])

    # -----------------------------------------
    # Split the data into training & testing sets
    # -----------------------------------------
    print("[INFO] Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------------------
    # Train the model
    # -----------------------------------------
    print("[INFO] Training Logistic Regression model...")
    model = train_model(X_train, y_train)

    # -----------------------------------------
    # Evaluate model on test set
    # -----------------------------------------
    print("[INFO] Evaluating model...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("==========================\n")

    # -----------------------------------------
    # Save trained model
    # -----------------------------------------
    print("[INFO] Saving model...")
    
    # Ensure models/ folder exists
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

    save_model(model, MODEL_OUTPUT_PATH)

    print("[DONE] Model training complete!")
    print(f"Saved model to: {MODEL_OUTPUT_PATH}")


