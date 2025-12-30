"""
feature_engineering.py
------------------------------------
This file handles FEATURE EXTRACTION using TF-IDF.

Steps performed:
1. Load cleaned text column
2. Convert text into numerical features using TF-IDF Vectorizer
3. Split dataset into training and testing sets
4. Return:
    - X_train, X_test
    - y_train, y_test
    - vectorizer (saved later with model)

Important:
TF-IDF helps convert text into meaningful numerical values for ML algorithms.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------
# Function: load_cleaned_dataset
# ----------------------------------------------------------------------
def load_cleaned_dataset(path="data/processed/cleaned_blogs.csv"):
    """
    Loads the cleaned dataset produced by preprocess.py.

    Returns:
        DataFrame if file exists, None if missing.
    """

    if not os.path.exists(path):
        print(f"[ERROR] Cleaned dataset not found at: {path}")
        print("[HINT] Run preprocess.py first to generate cleaned_blogs.csv")
        return None

    print(f"[INFO] Loading cleaned dataset from: {path}")
    df = pd.read_csv(path)

    print("[INFO] Cleaned dataset loaded successfully.")
    print(f"[INFO] Dataset shape: {df.shape}")
    return df


# ----------------------------------------------------------------------
# Function: apply_tfidf
# ----------------------------------------------------------------------
def apply_tfidf(df):
    """
    Converts cleaned text into TF-IDF vectors.

    Steps:
    - Initialize TfidfVectorizer
    - Fit to training text
    - Transform text into vectors

    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """

    print("[INFO] Starting TF-IDF vectorization...")

    # Extract features (text) and labels (categories)
    X = df["Cleaned_Text"]
    y = df["Labels"]

    # Initialize TF-IDF vectorizer
    # max_features limits the vocabulary to prevent memory overload
    vectorizer = TfidfVectorizer(max_features=5000)

    # Fit + transform TF-IDF
    X_tfidf = vectorizer.fit_transform(X)

    print("[INFO] TF-IDF vectorization complete.")
    print(f"[INFO] TF-IDF shape: {X_tfidf.shape}")  # (rows, features)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf,
        y,
        test_size=0.2,      # 20% for testing
        random_state=42,
        stratify=y          # keeps class distribution same
    )

    print("[INFO] Train-Test split complete.")
    print(f"[INFO] Training samples: {X_train.shape[0]}")
    print(f"[INFO] Testing samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, vectorizer


# ----------------------------------------------------------------------
# Self-Test Block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running feature_engineering.py directly...")

    # Load cleaned dataset
    df_clean = load_cleaned_dataset()

    if df_clean is not None:
        # Apply TF-IDF
        X_train, X_test, y_train, y_test, vectorizer = apply_tfidf(df_clean)

        print("[TEST] Feature engineering test completed successfully.")
