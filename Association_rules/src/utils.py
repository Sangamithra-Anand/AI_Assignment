"""
utils.py
Helper functions used across the project.
Now all logs are saved to output/logs/
"""

import os
import time


def ensure_folders_exist(folder_list):
    print("[INFO] Checking required folders...")

    for folder in folder_list:
        os.makedirs(folder, exist_ok=True)
        print(f"[INFO] Folder ready: {folder}")

    print("[INFO] All necessary folders are ready.\n")


def start_timer():
    return time.time()


def end_timer(start, step_name="Operation"):
    duration = time.time() - start
    print(f"[INFO] {step_name} completed in {duration:.2f} seconds.\n")


def log_message(message, log_file="output/logs/run_log.txt"):
    """
    Write logs into output/logs/ instead of main directory.
    """
    os.makedirs("output/logs", exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

    print(message)


def summarize_dataset(df):
    """
    Print dataset summary.
    """

    print("[INFO] Generating dataset summary...\n")

    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
    }

    print(f"  Rows: {summary['rows']}")
    print(f"  Columns: {summary['columns']}")
    print("  Missing values per column:")
    for col, val in summary["missing_values"].items():
        print(f"    {col}: {val}")

    print()
    return summary


