# load_data.py
# --------------------------------------------
# This file handles loading the exchange rate dataset.
# It reads the CSV file, converts the 'date' column into a
# proper datetime index, and returns a clean DataFrame.
# --------------------------------------------

import pandas as pd   # pandas is used for data handling
import os             # used to check file existence


def load_exchange_rate(filepath="data/exchange_rate.csv"):
    """
    Loads the exchange rate dataset from a CSV file.

    Parameters:
        filepath (str): Path to the dataset.
                        Default is "data/exchange_rate.csv".

    Returns:
        df (DataFrame): Cleaned dataset with 'date' as datetime index.

    Steps performed:
    1. Check if the file exists.
    2. Read the CSV using pandas.
    3. Convert 'date' column to datetime format.
    4. Set 'date' as index so it becomes a time-series.
    """

    # ----- 1. Check if file exists -----
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"[ERROR] File not found: {filepath}")

    # ----- 2. Read the CSV -----
    # Your CSV has column names: "date" and "Ex_rate"
    df = pd.read_csv(filepath)

    # ----- 3. Convert 'date' to datetime -----
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    # ----- 4. Set 'date' as index -----
    df = df.set_index("date")

    # ----- 5. Sort by date (important for time-series) -----
    df = df.sort_index()

    # ----- 6. Print basic info -----
    print("[INFO] Dataset loaded successfully.")
    print(f"[INFO] Number of rows: {len(df)}")
    print("[INFO] Columns:", list(df.columns))

    return df


