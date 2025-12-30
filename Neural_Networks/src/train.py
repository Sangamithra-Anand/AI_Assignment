"""
train.py
--------
This file is responsible for training a BASELINE Artificial Neural Network (ANN)
on the processed Alphabets dataset.

Main tasks:
1. Load the processed dataset from data/processed/
2. Split data into train and test sets
3. Convert labels (y) into numeric classes and one-hot vectors
4. Build an ANN model (using model_builder.py)
5. Train the model and evaluate it on the test set
6. Save:
   - The trained baseline model to models/baseline_model.h5
   - Basic metrics (accuracy, loss) to output/metrics/baseline_metrics.json
   - A simple training log to reports/training_log.txt

NOTE:
This file expects that:
- preprocess.py has already created the processed CSV
- model_builder.py (which we'll create next) has a function: build_ann_model(...)
"""

import os
import json
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf  # Keras lives inside TensorFlow

# Import configuration values from config.py
from config import (
    PROCESSED_DATA_PATH,   # Where the processed CSV is saved
    TARGET_COLUMN,         # Name of the label column (e.g., "label")
    TEST_SIZE,             # Fraction of data used for testing (e.g., 0.2)
    RANDOM_STATE,          # Seed for reproducibility
    TRAIN_PARAMS,          # Default training parameters (epochs, batch_size, etc.)
    MODELS_DIR,            # Folder to save .h5 models
    TRAINING_LOG_PATH,     # Path for training log file
    BASELINE_METRICS_PATH, # Path for baseline_metrics.json
    create_directories     # Helper to ensure folders exist
)

# We will build the ANN model in a separate file model_builder.py
# For now, we just assume it exists and has build_ann_model().
from model_builder import build_ann_model


def load_processed_data() -> pd.DataFrame:
    """
    Load the fully processed dataset from the CSV file.

    Why this function:
    - To centralize loading logic
    - To give clean error messages if the file does not exist
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        # Inform the user what to run if file is missing.
        raise FileNotFoundError(
            f"[ERROR] Processed data file not found at: {PROCESSED_DATA_PATH}\n"
            "Please run 'preprocess.py' first to generate the processed dataset."
        )

    print(f"[INFO] Loading processed dataset from: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"[INFO] Processed dataset shape: {df.shape}")
    return df


def split_features_and_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split DataFrame into features (X) and target labels (y), and convert y to integers.

    Steps:
    - Separate X = all columns except TARGET_COLUMN
    - y = TARGET_COLUMN
    - Convert y to integer class labels using pandas.factorize (if not already int)

    Returns:
        X_values: numpy array of features
        y_int: numpy array of integer labels (0, 1, 2, ... n_classes-1)
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"[ERROR] Target column '{TARGET_COLUMN}' not found in processed dataset.\n"
            "Check TARGET_COLUMN in config.py."
        )

    # X = all feature columns (data for the model)
    X = df.drop(columns=[TARGET_COLUMN])
    # y_raw = original labels (might be strings, chars, etc.)
    y_raw = df[TARGET_COLUMN]

    print(f"[INFO] Number of feature columns: {X.shape[1]}")

    # If labels are not numeric, convert them to integer codes:
    # factorize returns (array_of_codes, unique_values_array)
    # Example: ["A", "B", "C"] → [0, 1, 2]
    y_int, label_classes = pd.factorize(y_raw)

    print("[INFO] Label encoding complete.")
    print("[INFO] Classes mapping (index -> label):")
    for idx, label in enumerate(label_classes):
        print(f"   {idx} -> {label}")

    # Optionally, you could save this mapping to a file if you want later.
    # For now, we just print it.

    # Convert X to numpy array for Keras
    X_values = X.values

    return X_values, y_int


def train_test_split_data(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into train and test sets using sklearn's train_test_split.

    Why:
    - To evaluate how well the model generalizes to unseen data.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,      # e.g., 0.2 = 20% test
        random_state=RANDOM_STATE,
        stratify=y  # ensures same label distribution in train & test
    )

    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] X_test  shape: {X_test.shape}")
    print(f"[INFO] y_train length: {len(y_train)}")
    print(f"[INFO] y_test  length: {len(y_test)}")

    return X_train, X_test, y_train, y_test


def convert_labels_to_one_hot(y: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Convert integer labels to one-hot encoded vectors for Keras.

    Example:
        y = [0, 2, 1]
        num_classes = 3
        y_one_hot =
        [
          [1, 0, 0],
          [0, 0, 1],
          [0, 1, 0]
        ]

    Why:
    - For multi-class classification, Keras with 'categorical_crossentropy'
      expects one-hot encoded labels.

    Returns:
        y_one_hot: one-hot encoded labels
        num_classes: number of unique classes
    """
    # Number of distinct classes in y
    num_classes = len(np.unique(y))

    # Keras utility function to convert integer labels to one-hot vectors
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    print(f"[INFO] Number of classes: {num_classes}")
    print(f"[INFO] y one-hot shape: {y_one_hot.shape}")

    return y_one_hot, num_classes


def save_training_log(history: tf.keras.callbacks.History) -> None:
    """
    Save training history (loss and accuracy per epoch) to a text file.

    Args:
        history: The History object returned by model.fit()
    """
    # Extract history dictionary (contains lists for each metric per epoch)
    hist = history.history

    # Ensure directory exists for the log file
    os.makedirs(os.path.dirname(TRAINING_LOG_PATH), exist_ok=True)

    print(f"[INFO] Saving training log to: {TRAINING_LOG_PATH}")
    with open(TRAINING_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("Training History (per epoch):\n")
        for i in range(len(hist["loss"])):
            # For each epoch, write loss and accuracy
            loss = hist["loss"][i]
            acc = hist.get("accuracy", [None])[i]
            val_loss = hist.get("val_loss", [None])[i]
            val_acc = hist.get("val_accuracy", [None])[i]
            f.write(
                f"Epoch {i+1}: "
                f"loss={loss:.4f}, acc={acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}\n"
            )


def save_baseline_metrics(metrics: Dict[str, Any]) -> None:
    """
    Save baseline evaluation metrics (loss, accuracy, etc.) as JSON.

    Args:
        metrics: Dictionary with metric names and values.
    """
    # Ensure directory exists for metrics JSON
    os.makedirs(os.path.dirname(BASELINE_METRICS_PATH), exist_ok=True)

    print(f"[INFO] Saving baseline metrics to: {BASELINE_METRICS_PATH}")
    with open(BASELINE_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def train_baseline_model() -> None:
    """
    Full pipeline for training the baseline ANN model.

    Steps:
    1. Load processed data
    2. Split into train and test
    3. Convert y to one-hot
    4. Build the model (using model_builder.build_ann_model)
    5. Train the model
    6. Evaluate on test data
    7. Save model, metrics, and training log
    """

    # -------------------------
    # 1. Load processed dataset
    # -------------------------
    df_processed = load_processed_data()

    # -------------------------
    # 2. Split into X and y
    # -------------------------
    X, y_int = split_features_and_target(df_processed)

    # -------------------------
    # 3. Train-test split
    # -------------------------
    X_train, X_test, y_train_int, y_test_int = train_test_split_data(X, y_int)

    # -------------------------
    # 4. Convert labels to one-hot vectors for Keras
    # -------------------------
    y_train, num_classes = convert_labels_to_one_hot(y_train_int)
    y_test, _ = convert_labels_to_one_hot(y_test_int)

    # -------------------------
    # 5. Build the ANN model
    # -------------------------
    # Input dimension = number of features per sample
    input_dim = X_train.shape[1]

    print("[INFO] Building baseline ANN model...")
    # We assume build_ann_model is implemented in model_builder.py.
    # It should accept input_dim, num_classes, and TRAIN_PARAMS.
    model = build_ann_model(
        input_dim=input_dim,
        num_classes=num_classes,
        train_params=TRAIN_PARAMS
    )

    model.summary()  # prints the model architecture

    # -------------------------
    # 6. Train the model
    # -------------------------
    print("[INFO] Starting model training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),          # monitor performance on test set
        epochs=TRAIN_PARAMS.epochs,                # number of passes over entire dataset
        batch_size=TRAIN_PARAMS.batch_size,        # how many samples per gradient update
        verbose=1
    )

    # -------------------------
    # 7. Evaluate the model on test data
    # -------------------------
    print("[INFO] Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"[RESULT] Test Loss    : {test_loss:.4f}")
    print(f"[RESULT] Test Accuracy: {test_accuracy:.4f}")

    # -------------------------
    # 8. Save the trained baseline model
    # -------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    baseline_model_path = os.path.join(MODELS_DIR, "baseline_model.h5")

    print(f"[INFO] Saving baseline model to: {baseline_model_path}")
    model.save(baseline_model_path)

    # -------------------------
    # 9. Save metrics and training log
    # -------------------------
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs": TRAIN_PARAMS.epochs,
        "batch_size": TRAIN_PARAMS.batch_size,
        "learning_rate": TRAIN_PARAMS.learning_rate,
    }
    save_baseline_metrics(metrics)
    save_training_log(history)

    print("[INFO] Baseline training pipeline completed successfully.")


# ---------------------------------------------------------------------
# Entry point: if user runs `python train.py` from terminal.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """
    When this file is run directly, we:
    1. Create all necessary directories.
    2. Train the baseline ANN model.
    """
    # Ensure all folders used in config exist
    create_directories()

    # Run the baseline training pipeline
    try:
        train_baseline_model()
    except FileNotFoundError as e:
        # Most common issue: processed CSV missing.
        print(e)


