"""
utils.py
---------
This file contains helper/utility functions used across the project.

Why utils.py?
-------------
Instead of writing the same code (folder creation, logging, timers)
in every file, we store them here once and reuse everywhere.

Contains:
1. ensure_folder_exists() → Auto-create directories
2. log_message() → Simple logging tool
3. start_timer() & end_timer() → Measure runtime of tasks
"""

import os
import time


# -------------------------------------------------------------------------
# 1. Ensure a folder exists (AUTO-create if missing)
# -------------------------------------------------------------------------
def ensure_folder_exists(path):
    """
    Creates a folder if it does not exist.

    Parameters:
    -----------
    path : str
        Path of folder to create.

    Why?
    ----
    Prevents FileNotFoundError when saving reports, models, plots, etc.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[AUTO] Created folder: {path}")


# -------------------------------------------------------------------------
# 2. Logging helper
# -------------------------------------------------------------------------
def log_message(message, log_file="logs/project.log"):
    """
    Writes a message to a log file and prints it to console.

    Parameters:
    -----------
    message : str
        Text to log
    log_file : str
        Path to log file

    Explanation:
    ------------
    - Creates logs folder automatically
    - Appends logs with timestamps
    - Helps track pipeline progress or debug issues
    """
    ensure_folder_exists("logs")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a") as file:
        file.write(f"[{timestamp}] {message}\n")

    print(f"[LOG] {message}")


# -------------------------------------------------------------------------
# 3. Timer start
# -------------------------------------------------------------------------
def start_timer():
    """
    Starts a timer and returns the start time.

    Usage:
    -------
    t = start_timer()
    ... your task ...
    end_timer(t, "Task Name")
    """
    return time.time()


# -------------------------------------------------------------------------
# 4. Timer end
# -------------------------------------------------------------------------
def end_timer(start_time, task_name="Task"):
    """
    Ends the timer and prints how long a task took.

    Parameters:
    -----------
    start_time : float
        Output of start_timer()
    task_name : str
        Name of the completed task
    """
    end_time = time.time()
    duration = end_time - start_time
    print(f"[TIMER] {task_name} completed in {duration:.2f} seconds.")


# -------------------------------------------------------------------------
# TEST BLOCK — run with:
# python src/utils.py
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running utils.py directly...")

    try:
        ensure_folder_exists("test_folder")
        log_message("Utils.py test successful!")

        t = start_timer()
        time.sleep(1.2)  # simulate work
        end_timer(t, "Sample Timer Test")

        print("\n[TEST] utils.py is working correctly ✔️")
    except Exception as e:
        print(f"[TEST ERROR] {e}")
