import os
import pandas as pd

# -------------------------------------------------------------
# Function: create_directories
# Purpose : Automatically create required folders for the project.
#           These folders store processed data, reports, and output files.
# -------------------------------------------------------------
def create_directories():
    """
    Create required directories if they don't exist.
    This prevents errors later when saving processed files.
    """
    os.makedirs("data/processed", exist_ok=True)  # Stores cleaned dataset
    os.makedirs("data/reports", exist_ok=True)    # Stores EDA summary/report files
    os.makedirs("output", exist_ok=True)          # Stores plots and output files


# -------------------------------------------------------------
# Function: load_dataset
# Purpose : Load the raw Adult dataset from the given file path.
#           Also calls create_directories() to ensure the folder
#           structure exists before any processing is done.
# -------------------------------------------------------------
def load_dataset(path="data/raw/adult_with_headers.csv"):
    """
    Loads the Adult dataset from the given path.

    Parameters:
        path (str): File path to the CSV file.

    Returns:
        df (DataFrame): Loaded dataset as a pandas DataFrame.
    """

    # Ensure required folders exist before loading or saving files
    create_directories()

    # Check whether the dataset file actually exists
    if not os.path.exists(path):
        # If not found, raise a clear error with the exact missing path
        raise FileNotFoundError(f"Dataset not found at: {path}")

    print(f"[INFO] Loading dataset from: {path}")

    # pandas reads the CSV file and converts it into a DataFrame
    df = pd.read_csv(path)

    print(f"[INFO] Dataset loaded successfully. Shape: {df.shape}")

    # Return the loaded DataFrame so other modules can use it
    return df
