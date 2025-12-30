"""
config.py
---------
Central configuration file for the Alphabet ANN project.

This file:
- Defines important folder paths (data, models, reports, outputs)
- Defines dataset-related settings (file name, target column)
- Sets training & model hyperparameters (batch size, epochs, etc.)
- Provides a helper function to auto-create all required folders

You can import these values anywhere using:
    from config import (
        BASE_DIR, RAW_DATA_PATH, PROCESSED_DATA_PATH,
        MODELS_DIR, REPORTS_DIR, OUTPUT_DIR,
        TRAIN_PARAMS, HYPERPARAM_GRID, create_directories
    )
"""

import os
from dataclasses import dataclass

# ------------------------------------------------------------
# 1. BASE PROJECT PATHS
# ------------------------------------------------------------
# __file__ = path to this file (config.py)
# os.path.dirname(__file__)        -> src/
# os.path.dirname(os.path.dirname(__file__)) -> project root (alphabet_ann_project/)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Main sub-folders under the project root
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# ------------------------------------------------------------
# 2. DATASET CONFIGURATION
# ------------------------------------------------------------

# CSV file name inside data/raw/
# NOTE: make sure the file name exactly matches the one in your folder.
RAW_DATA_FILENAME = "Alphabets_data.csv"

# Full path to the raw CSV file
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, RAW_DATA_FILENAME)

# Processed dataset file name (after cleaning, scaling, encoding, etc.)
PROCESSED_DATA_FILENAME = "alphabets_final_processed.csv"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME)

# Name of the target / label column in the dataset.
# ⚠️ IMPORTANT: Change this if your CSV uses a different name (for example: "alphabet", "class", "letter")
TARGET_COLUMN = "letter"

# ------------------------------------------------------------
# 3. TRAIN / TEST / VALIDATION SPLIT SETTINGS
# ------------------------------------------------------------

# Random seed for reproducibility (any fixed integer is fine)
RANDOM_STATE = 42

# Percentage of data used for testing (e.g. 0.2 = 20% test, 80% train)
TEST_SIZE = 0.2

# Optional: fraction of training data to use as validation set after the split
# Example: 0.1 means 10% of the training split will be used as validation
VALIDATION_SPLIT = 0.1

# ------------------------------------------------------------
# 4. DEFAULT TRAINING HYPERPARAMETERS (BASELINE MODEL)
# ------------------------------------------------------------

@dataclass
class TrainParams:
    """
    Training parameters for the baseline ANN model.
    You can import and pass this to your training function.
    """
    batch_size: int = 32          # How many samples per gradient update
    epochs: int = 50              # How many passes over the full dataset
    learning_rate: float = 0.001  # Step size for optimizer
    hidden_units: int = 64        # Neurons in the main hidden layer
    hidden_layers: int = 1        # How many hidden layers to use
    activation: str = "relu"      # Activation function for hidden layers
    output_activation: str = "softmax"  # For multi-class classification


# Create a default instance used in train.py
TRAIN_PARAMS = TrainParams()

# ------------------------------------------------------------
# 5. HYPERPARAMETER GRID FOR TUNING
# ------------------------------------------------------------
# This grid is used by tune_hyperparameters.py
# You can expand or shrink this grid based on time and performance.
HYPERPARAM_GRID = {
    # number of hidden layers to try
    "hidden_layers": [1, 2],

    # number of neurons in each hidden layer
    "hidden_units": [32, 64, 128],

    # batch sizes to test
    "batch_size": [32, 64],

    # learning rates to test
    "learning_rate": [0.001, 0.0005],

    # activation functions to test for hidden layers
    "activation": ["relu", "tanh"]
}

# ------------------------------------------------------------
# 6. FILE NAMES FOR REPORTS / LOGS
# ------------------------------------------------------------

# Where to store training logs and hyperparameter search results
TRAINING_LOG_PATH = os.path.join(REPORTS_DIR, "training_log.txt")
HYPERPARAM_RESULTS_PATH = os.path.join(REPORTS_DIR, "hyperparameter_search_results.csv")

# Metrics JSON files (baseline vs tuned)
BASELINE_METRICS_PATH = os.path.join(METRICS_DIR, "baseline_metrics.json")
TUNED_METRICS_PATH = os.path.join(METRICS_DIR, "tuned_metrics.json")

# Plot file paths
LOSS_CURVE_PATH = os.path.join(FIGURES_DIR, "loss_curve.png")
ACCURACY_CURVE_PATH = os.path.join(FIGURES_DIR, "accuracy_curve.png")
CONFUSION_MATRIX_PATH = os.path.join(FIGURES_DIR, "confusion_matrix.png")

# ------------------------------------------------------------
# 7. DIRECTORY CREATION HELPER
# ------------------------------------------------------------

def create_directories() -> None:
    """
    Create all directories required by the project.

    This function should be called once at the beginning
    of your main script (e.g. in main.py).

    Example usage:
        from config import create_directories
        create_directories()
    """
    dirs_to_create = [
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        OUTPUT_DIR,
        METRICS_DIR,
        FIGURES_DIR,
    ]

    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        # Note: exist_ok=True means "do not raise error if folder already exists"

