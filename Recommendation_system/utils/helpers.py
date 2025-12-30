"""
helpers.py
-----------
This file contains GENERIC UTILITY FUNCTIONS that can be reused
across the entire Anime Recommendation project.

What is included:
1. ensure_folder(path)
      - Safely create folders if they don't exist.

2. log_message(message, level="INFO")
      - Standardized console logging.

3. start_timer() and end_timer()
      - Simple timing helpers to measure how long steps take.

You can import these in any file like:
    from utils.helpers import ensure_folder, log_message, start_timer, end_timer
"""

import os
import time
from datetime import datetime


# -------------------------------------------------------------
# Function: ensure_folder
# Purpose : Create a folder if it does not already exist.
# -------------------------------------------------------------
def ensure_folder(path: str):
    """
    Ensures that a folder exists.

    Parameters:
        path (str): Folder path to check/create.

    Behavior:
    - If the folder does not exist, it will be created.
    - If it already exists, nothing happens.

    Example:
        ensure_folder("data/processed")
    """
    if not path:  # Avoid issues if empty string is passed
        return

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Created folder: {path}")


# -------------------------------------------------------------
# Function: log_message
# Purpose : Simple, unified logging to console.
# -------------------------------------------------------------
def log_message(message: str, level: str = "INFO"):
    """
    Prints a formatted log message with timestamp and level.

    Parameters:
        message (str): The message to log.
        level (str): Log level text (e.g., "INFO", "ERROR", "WARNING").

    Example:
        log_message("Loading dataset...", "INFO")
        log_message("File not found!", "ERROR")
    """

    # Current timestamp in a readable format
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Final log format: [2025-12-05 04:21:11] [INFO] Your message here
    print(f"[{timestamp}] [{level}] {message}")


# -------------------------------------------------------------
# Function: start_timer
# Purpose : Start timing a block of code.
# -------------------------------------------------------------
def start_timer():
    """
    Starts a simple timer using time.perf_counter().

    Returns:
        float: A high-precision start time value.

    Usage:
        start = start_timer()
        # ... run some code ...
        end_timer(start, "Data loading")
    """
    return time.perf_counter()


# -------------------------------------------------------------
# Function: end_timer
# Purpose : End timing and print elapsed time.
# -------------------------------------------------------------
def end_timer(start_time: float, task_name: str = "Task"):
    """
    Ends a timer started with start_timer() and prints how long it took.

    Parameters:
        start_time (float): The value returned by start_timer()
        task_name (str): Name of the task being measured (for display)

    Example:
        start = start_timer()
        # ... your code ...
        end_timer(start, "Preprocessing")
    """
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"[TIMER] {task_name} completed in {elapsed:.2f} seconds.")

