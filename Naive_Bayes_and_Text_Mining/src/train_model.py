"""
train_model.py
------------------------------------
This file trains the Naive Bayes text classification model.

Steps performed:
1. Load cleaned dataset using feature_engineering.py
2. Apply TF-IDF vectorization
3. Train Multinomial Naive Bayes model
4. Save trained model + TF-IDF vectorizer into models/ folder

The model will be used later for evaluation and predictions.
"""

import os
import pickle
from sklearn.naive_bayes import MultinomialNB

from feature_engineering import load_cleaned_dataset, apply_tfidf


# ----------------------------------------------------------------------
# Helper: ensure models folder exists
# ----------------------------------------------------------------------
def ensure_models_folder(path="models/"):
    """
    Creates 'models/' folder if it does not exist.
    Prevents save errors when storing the trained model.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[AUTO] Created folder: {path}")


# ----------------------------------------------------------------------
# Main function: Train Naive Bayes model
# ----------------------------------------------------------------------
def train_naive_bayes():
    """
    Trains the Naive Bayes classifier using TF-IDF features.

    Returns:
        model (trained Naive Bayes classifier)
        X_train, X_test, y_train, y_test
    """

    print("[INFO] Loading cleaned dataset...")
    df = load_cleaned_dataset()

    if df is None:
        return None, None, None, None, None

    print("[INFO] Applying TF-IDF...")
    X_train, X_test, y_train, y_test, vectorizer = apply_tfidf(df)

    print("[INFO] Training Naive Bayes model...")
    model = MultinomialNB()

    # Fit the model on training data
    model.fit(X_train, y_train)

    print("[INFO] Model training completed successfully.")

    # Save model + vectorizer together
    ensure_models_folder()

    model_path = "models/naive_bayes_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"[INFO] Saved trained model: {model_path}")
    print(f"[INFO] Saved TF-IDF vectorizer: {vectorizer_path}")

    return model, X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# Self-Test Block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running train_model.py directly...")
    model, X_train, X_test, y_train, y_test = train_naive_bayes()

    if model is not None:
        print("[TEST] Training test completed successfully.")
