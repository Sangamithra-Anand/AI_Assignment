"""
load_data.py
-------------
This upgraded version handles messy Excel files that include
text descriptions above the actual data table.

New Features:
-------------
1. Detects the first numeric row (start of real dataset)
2. Ignores description/documentation rows
3. Removes unnamed/empty columns
4. Renames numeric headers into proper Glass dataset column names
5. Loads a clean DataFrame ready for EDA & preprocessing
"""

import os
import pandas as pd


def load_glass_data(path="data/raw/glass.xlsx"):
    """
    Loads the Glass dataset even if the Excel file contains
    extra text rows before the actual numeric table.

    Steps:
    ------
    1. Load file without headers
    2. Detect the row where numeric data starts
    3. Reload file using that row as header
    4. Clean unnecessary columns
    5. Rename columns correctly
    """

    # ----------------------------------------
    # 1. Validate file existence
    # ----------------------------------------
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] Raw dataset not found at: {path}\n"
            f"Place glass.xlsx inside data/raw/"
        )

    # ----------------------------------------
    # 2. Load file without headers to inspect raw rows
    # ----------------------------------------
    raw_df = pd.read_excel(path, header=None)
    print("[INFO] Raw Excel file loaded. Detecting actual data rows...")

    # ----------------------------------------
    # 3. Detect numeric table starting row
    # ----------------------------------------
    numeric_start = None

    for i in range(len(raw_df)):
        row = raw_df.iloc[i]

        # Count numeric-like entries in row
        numeric_count = sum(
            str(x).replace(".", "", 1).isdigit() for x in row
        )

        # If row has at least 5 numeric values → it's real data
        if numeric_count >= 5:
            numeric_start = i
            break

    if numeric_start is None:
        raise ValueError("[ERROR] Could not detect numeric data in Excel file.")

    print(f"[INFO] Numeric data detected starting at Excel row: {numeric_start}")

    # ----------------------------------------
    # 4. Load again using detected row as header
    # ----------------------------------------
    df = pd.read_excel(path, header=numeric_start)

    # ----------------------------------------
    # 5. Remove unnamed/empty columns
    # ----------------------------------------
    df = df.loc[:, ~df.columns.astype(str).str.contains("Unnamed")]

    # ----------------------------------------
    # 6. Convert all column names to strings
    # ----------------------------------------
    df.columns = df.columns.astype(str)

    # ----------------------------------------
    # 7. Rename numeric column names to actual Glass dataset headers
    # ----------------------------------------
    correct_columns = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]

    if len(df.columns) == 10:
        print("[INFO] Renaming column names to standard Glass dataset headers...")
        df.columns = correct_columns
    else:
        print("[WARNING] Unexpected column count. Column renaming skipped.")

    # ----------------------------------------
    # 8. Print clean dataset info
    # ----------------------------------------
    print("\n[INFO] Cleaned Glass dataset loaded successfully!")
    print(f"[INFO] Dataset shape: {df.shape}")
    print("[INFO] First 5 rows:")
    print(df.head())

    return df


# Optional: Test this script independently
if __name__ == "__main__":
    print("[TEST] Running load_data.py directly...")

    try:
        df = load_glass_data()
        print("\n[TEST] load_data.py is working correctly ✔️")
    except Exception as e:
        print(f"[TEST ERROR] {e}")
