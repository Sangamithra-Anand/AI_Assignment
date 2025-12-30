# utils.py
# ------------------------------
# This file contains helper functions used across the project.
# It mainly handles automatic folder creation and plot-saving utilities.
# ------------------------------

import os   # 'os' library lets us work with folders and file paths


def create_directories():
    """
    Automatically creates ALL required project folders.

    Why we do this:
    - The project needs folders like models/, results/, plots/, forecasts/
    - Instead of creating these manually, the code will auto-generate them.
    - If a folder already exists, 'exist_ok=True' prevents any errors.
    """

    # List of required folder paths
    folders = [
        "models",                # Store trained model files (.pkl)
        "results",               # Main results folder
        "results/plots",         # All visualizations and graphs
        "results/metrics",       # Evaluation metrics stored here
        "results/forecasts"      # Forecast CSV files stored here
    ]

    # Create each folder in the list
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        # 'exist_ok=True' means:
        # → If folder does NOT exist → create it
        # → If folder already exists → do nothing (no error)

    print("[INFO] Required folders are created or already exist.")


def save_plot(path):
    """
    Ensures the folder exists before saving a plot.

    Example:
        save_plot("results/plots/time_series.png")

    Why this function exists:
    - Sometimes you save a plot into a folder that was not created yet.
    - This function prevents errors by creating the folder automatically.
    """

    # Extract folder path from file path
    folder = os.path.dirname(path)

    # Create the folder (if not existing)
    os.makedirs(folder, exist_ok=True)

    # This function does NOT save the plot itself.
    # It only ensures the folder exists so matplotlib can save the image safely.


