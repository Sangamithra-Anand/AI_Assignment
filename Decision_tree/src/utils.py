"""
utils.py
---------
This file contains UTILITY FUNCTIONS used across the project.

What this script provides:
✔ Auto-create all required project folders
✔ Simple logging function (writes to logs/project.log)
✔ Timer utilities to measure pipeline runtime

Every function is fully explained inside the code.
"""

import os
import time


# ============================================================
# 1. AUTO-CREATE PROJECT FOLDERS
# ============================================================

def create_project_folders():
    """
    Automatically creates all folders needed for the project.

    This prevents errors like:
        FileNotFoundError: directory does not exist

    Called once at the start of the pipeline (inside main.py).
    """

    folders = [
        "data",
        "data/raw",
        "data/processed",
        "models",
        "reports",
        "outputs",
        "logs"
    ]

    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"[AUTO] Ensured folder exists: {folder}")
        except Exception as e:
            print(f"[ERROR] Could not create folder '{folder}': {e}")


# ============================================================
# 2. LOGGING SYSTEM
# ============================================================

def log_message(message):
    """
    Write messages to logs/project.log for debugging and tracking.

    Parameters:
    -----------
    message : str
        Text to log with timestamp.

    Example:
        log_message("[INFO] Training model completed")
    """

    logs_path = "logs"
    os.makedirs(logs_path, exist_ok=True)

    logfile = os.path.join(logs_path, "project.log")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(logfile, "a") as f:
            f.write(f"{timestamp}  {message}\n")
    except Exception as e:
        print(f"[ERROR] Could not write to log file: {e}")


# ============================================================
# 3. TIMER UTILITY
# ============================================================

def start_timer():
    """
    Starts a timer and returns the current time.
    Used to measure execution time of any process.
    """
    return time.time()


def end_timer(start_time, message="Process"):
    """
    Calculates how long a process took and prints/returns the elapsed time.

    Parameters:
    -----------
    start_time : float
        Value returned from start_timer()

    message : str
        Name of the process being measured
    """

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"[TIMER] {message} completed in {elapsed:.2f} seconds")
    log_message(f"[TIMER] {message} took {elapsed:.2f} seconds")

    return elapsed


# ============================================================
# 4. SELF-TEST MODE
# ============================================================

if __name__ == "__main__":
    print("[TEST] Running utils.py directly...")

    # Test folder creation
    create_project_folders()

    # Test logging
    log_message("[TEST] utils.py logging system is working")

    # Test timer
    t0 = start_timer()
    time.sleep(1)  # simulate a delay
    end_timer(t0, "Test Timer")

    print("[TEST] utils.py test completed.")
