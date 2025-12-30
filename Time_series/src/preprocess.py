# preprocess.py
# ----------------------------------------------------------
# This file handles all data cleaning steps such as:
# - Handling missing values
# - Removing anomalies (optional basic filtering)
# - Ensuring the data is ready for modeling
#
# These preprocessing steps ensure that ARIMA and 
# Exponential Smoothing models receive clean input.
# ----------------------------------------------------------

import pandas as pd


def handle_missing_values(df):
    """
    Handles missing values in the dataset.

    Why this function exists:
    - Time-series models cannot work properly with missing values.
    - Some datasets skip weekends or holidays.
    - We fill missing values using forward-fill (last known value).

    Parameters:
        df (DataFrame): Input dataset

    Returns:
        df (DataFrame): Dataset with no missing values
    """

    # Check if missing values exist
    if df.isnull().sum().sum() > 0:
        print("[INFO] Missing values found. Applying forward-fill...")

        # Forward-fill:
        # Example: If 01 Jan value is missing, take 31 Dec value.
        df = df.ffill()

        # If front has missing values, use backward fill
        df = df.bfill()

    else:
        print("[INFO] No missing values detected.")

    return df


def remove_anomalies(df, column_name):
    """
    Removes extreme sudden spikes from the data.
    
    This is a very basic anomaly filter:
    - Any value beyond mean ± 3*std is considered abnormal.
    - This approach is optional and simple for demonstration.

    Parameters:
        df (DataFrame): Input dataset
        column_name (str): Name of the exchange rate column

    Returns:
        df (DataFrame): Smoothed dataset without extreme spikes.
    """

    # Calculate mean and standard deviation
    mean_val = df[column_name].mean()
    std_val = df[column_name].std()

    # Define upper and lower boundaries
    upper_limit = mean_val + 3 * std_val
    lower_limit = mean_val - 3 * std_val

    print(f"[INFO] Removing anomalies outside range: {lower_limit:.4f} to {upper_limit:.4f}")

    # Clamp values to these limits
    df[column_name] = df[column_name].clip(lower=lower_limit, upper=upper_limit)

    return df


def preprocess_data(df, column_name):
    """
    Full preprocessing pipeline:
    1. Handle missing values
    2. Remove anomalies

    Parameters:
        df (DataFrame): Raw dataset
        column_name (str): Name of the exchange rate column

    Returns:
        df (DataFrame): Cleaned and ready for modeling
    """

    print("[INFO] Starting preprocessing...")

    df = handle_missing_values(df)
    df = remove_anomalies(df, column_name)

    print("[INFO] Preprocessing completed.")

    return df


