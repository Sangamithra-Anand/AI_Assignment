import os
import pandas as pd

def load_dataset(path="data/raw/EastWestAirlines.xlsx"):
    """
    Loads the dataset from the Excel file.
    This version automatically detects which sheet contains numeric data.

    Returns:
        df_raw     → the raw DataFrame loaded from Excel
        df_numeric → the DataFrame containing only numeric columns
    """

    # ---------------------------------------------------------
    # 1. Check if file exists
    # ---------------------------------------------------------
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] Dataset not found at: {path}\n"
            "Make sure the Excel file is inside: data/raw/"
        )

    print(f"[INFO] Loading Excel file: {path}")

    # ---------------------------------------------------------
    # 2. List all sheets inside the Excel file
    # ---------------------------------------------------------
    excel_file = pd.ExcelFile(path)
    sheet_names = excel_file.sheet_names

    print("[INFO] Available sheets found:", sheet_names)

    # ---------------------------------------------------------
    # 3. Try each sheet until we find one with numeric data
    # ---------------------------------------------------------
    df_raw = None
    df_numeric = None

    for sheet in sheet_names:
        print(f"[INFO] Trying sheet: '{sheet}'")

        temp_df = pd.read_excel(path, sheet_name=sheet)

        # Take numeric columns
        numeric_df = temp_df.select_dtypes(include=['number'])

        if numeric_df.shape[1] > 0:
            print(f"[SUCCESS] Numeric data found in sheet: '{sheet}'")
            df_raw = temp_df
            df_numeric = numeric_df.copy()
            break  # stop after finding the correct sheet

    # ---------------------------------------------------------
    # 4. If no sheet contains numeric columns, throw an error
    # ---------------------------------------------------------
    if df_numeric is None:
        raise ValueError(
            "[ERROR] No sheet with numeric columns was found.\n"
            "Your Excel file may only contain description text.\n"
            "Please check the structure of the file."
        )

    print(f"[INFO] Raw dataset shape: {df_raw.shape}")
    print(f"[INFO] Numeric dataset shape: {df_numeric.shape}")
    print("[INFO] Numeric columns:", list(df_numeric.columns))

    return df_raw, df_numeric



# ---------------------------------------------------------
# TESTING BLOCK (runs only when executing this file directly)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[TEST] Running load_data.py directly...")
    try:
        df_raw, df_numeric = load_dataset()
        print("[TEST] load_data.py executed successfully.")
    except Exception as e:
        print("[TEST ERROR]", e)
