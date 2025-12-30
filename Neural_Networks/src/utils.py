"""
utils.py
--------
This file contains small helper functions that are reused across the project.

These utilities make the rest of the code cleaner and avoid repeating logic.

Included utilities:
1. print_section()         -> prints a clear visual section header
2. timer()                 -> decorator to measure how long a function takes
3. save_json()             -> save a dictionary to a JSON file
4. load_json()             -> load a JSON file safely
5. safe_mkdir()            -> safely create directories
"""

import os
import json
import time
from functools import wraps


def print_section(title: str) -> None:
    """
    Print a nicely formatted section title.
    Helps to visually separate output in the terminal.

    Example:
        print_section("Starting Training")

    Output:
    ---------------------- Starting Training ----------------------
    """
    print("\n" + "-" * 30 + f" {title} " + "-" * 30 + "\n")


def timer(func):
    """
    Decorator to measure how long a function takes to run.

    Usage:

        @timer
        def my_function():
            ...

    When called, it will print:
        [TIMER] my_function took 3.42 seconds
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()          # start time
        result = func(*args, **kwargs)
        end = time.time()            # end time
        elapsed = end - start
        print(f"[TIMER] {func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper


def save_json(data: dict, path: str) -> None:
    """
    Save any dictionary as a JSON file.

    Args:
        data: dictionary to save
        path: path to JSON file

    Automatically creates the folder if it doesn't exist.
    """
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"[INFO] Saved JSON to: {path}")


def load_json(path: str) -> dict:
    """
    Load JSON safely.

    If file doesn't exist, returns an empty dictionary instead of crashing.
    """
    if not os.path.exists(path):
        print(f"[WARNING] JSON file not found at: {path} (returning empty dict)")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mkdir(path: str) -> None:
    """
    Safely create a directory if it doesn’t exist.

    Unlike os.makedirs() directly, this function prints a confirmation message.
    """
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Ensured directory exists: {path}")


