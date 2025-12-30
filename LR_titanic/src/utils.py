"""
utils.py
--------
This file contains small helper functions used across the ML project.

These functions are NOT mandatory, but they help keep
preprocessing.py, train_model.py, and evaluate.py clean and readable.

This file mainly includes:

1. ensure_directory(path)
      → Creates a folder automatically if it doesn't exist
        (Used for /output/, /output/plots/, /output/reports/, /models/)

2. print_divider(title)
      → Prints pretty console titles for better readability

3. get_project_root()
      → Returns the ROOT folder of your project automatically
"""

import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def ensure_directory(path: str):
    """
    Creates a folder if it does NOT already exist.
    Prevents errors like:
        OSError: Cannot save file into a non-existent directory
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Created directory: {path}")


def print_divider(title: str = ""):
    """
    Prints a clear divider in the console.
    Helps to separate steps visually.
    """
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60 + "\n")


def get_project_root():
    """
    Automatically returns the root directory of your project.
    
    Example:
    If utils.py is at:
        project/src/utils.py

    This function returns:
        project/
    """
    return os.path.dirname(os.path.dirname(__file__))
