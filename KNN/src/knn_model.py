"""
knn_model.py
----------------------
This module builds, trains, tunes, and saves the K-Nearest Neighbors model.

Features:
    ✔ Train-test split (80/20)
    ✔ Choose K value (hyperparameter)
    ✔ Choose distance metric
    ✔ Train the KNN model
    ✔ Save the trained model

Everything is explained inside the code.
"""

import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# =========================================================
# FUNCTION: save_model()
# Purpose :
#     - Saves trained KNN model into output/models/
#     - Uses pickle for serialization
# =========================================================
def save_model(model, filename="knn_model.pkl"):
    os.makedirs("output/models", exist_ok=True)
    file_path = f"output/models/{filename}"

    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[INFO] Model saved: {file_path}")


# =========================================================
# FUNCTION: train_knn()
# Purpose :
#     - Trains a KNN model using Scikit-Learn.
#     - Allows choosing:
#         * K value (n_neighbors)
#         * Distance metric (metric = 'minkowski', 'euclidean', 'manhattan')
#
# Returns:
#     model          : Trained KNN classifier
#     X_test, y_test : Testing set for evaluation
# =========================================================
def train_knn(df, target_column="type", k=5, metric="minkowski"):
    print("\n[INFO] Starting KNN training...")

    # -------------------------------------------
    # Step 1: Separate input features and target
    # -------------------------------------------
    X = df.drop(columns=[target_column], errors="ignore")

    # Remove non-numeric columns (like animal name)
    X = X.select_dtypes(include=['int64', 'float64'])

    y = df[target_column]

    # -------------------------------------------
    # Step 2: Train-test split (80/20)
    # -------------------------------------------
    print("[INFO] Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Training size: {X_train.shape}")
    print(f"[INFO] Testing size:  {X_test.shape}")

    # -------------------------------------------
    # Step 3: Create the KNN model
    # -------------------------------------------
    print(f"[INFO] Training KNN model with K={k}, metric='{metric}'")

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric=metric
    )

    # -------------------------------------------
    # Step 4: Train model
    # -------------------------------------------
    model.fit(X_train, y_train)
    print("[INFO] KNN model training completed.")

    # -------------------------------------------
    # Step 5: Save model
    # -------------------------------------------
    save_model(model)

    return model, X_test, y_test


# =========================================================
# FUNCTION: find_best_k()
# Purpose :
#     - Tests multiple K values to find the best K for KNN.
#     - FIXED: Automatically removes non-numeric columns.
#       (Prevents “could not convert string to float: 'aardvark'” error)
#
# Returns:
#     best_k : Best-performing K value based on accuracy
# =========================================================
def find_best_k(df, target_column="type", k_values=None, metric="minkowski"):
    from sklearn.metrics import accuracy_score

    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 13]

    print("\n[INFO] Finding best K value...")

    # ----------------------------------------------------
    # Remove ALL non-numeric columns (fixes string errors)
    # ----------------------------------------------------
    X = df.drop(columns=[target_column], errors="ignore")
    X = X.select_dtypes(include=['int64', 'float64'])  # numeric only

    y = df[target_column]

    best_k = None
    best_score = -1

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric)

        # Train on full numeric dataset (for tuning)
        model.fit(X, y)

        predictions = model.predict(X)
        score = accuracy_score(y, predictions)

        print(f"[INFO] K={k} → Accuracy={score:.4f}")

        # Track best K
        if score > best_score:
            best_score = score
            best_k = k

    print(f"[RESULT] Best K = {best_k} (Accuracy = {best_score:.4f})")

    return best_k


# =========================================================
# TEST BLOCK
# Runs only when executing:
#     python src/knn_model.py
# =========================================================
if __name__ == "__main__":
    print("[TEST] Testing knn_model.py...")

    try:
        import pandas as pd

        df_sample = pd.read_csv("data/Zoo.csv")

        # Step 1: Find best K
        best_k = find_best_k(df_sample)
        print(f"[TEST] Best K found: {best_k}")

        # Step 2: Train full model
        model, X_test, y_test = train_knn(df_sample, k=best_k)

        print("[TEST] Training test completed.")

    except Exception as e:
        print("[TEST ERROR] Could not run training:", e)


