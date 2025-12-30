# visualize.py
# ---------------------------------------------------------------------
# This file contains visualization functions for:
#   1. Time series plot
#   2. Autocorrelation Function (ACF)
#   3. Partial Autocorrelation Function (PACF)
#
# These plots help in understanding:
#   - Trend
#   - Seasonality
#   - Lag correlations (important for ARIMA p, q selection)
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import statsmodels.api as sm
from utils import save_plot   # ensures folders exist before saving plots


def plot_time_series(df, column_name, output_path="results/plots/time_series.png"):
    """
    Plots the raw time series data.

    Why this plot?
    - Helps visualize long-term trend patterns
    - Identifies seasonal patterns
    - Shows sudden spikes/drops (anomalies)

    Parameters:
        df (DataFrame): Dataset
        column_name (str): Column containing the exchange rate
        output_path (str): Where to save the plot image
    """

    print("[INFO] Plotting time series...")

    # Ensure folder exists
    save_plot(output_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df[column_name], label="Exchange Rate", color="blue")
    plt.title("Exchange Rate Time Series")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()

    plt.savefig(output_path)
    plt.close()

    print(f"[INFO] Time series plot saved to {output_path}")


def plot_acf(df, column_name, output_path="results/plots/acf.png"):
    """
    Plots the Autocorrelation Function (ACF).

    Why ACF?
    - Shows correlation between the series and its lagged values.
    - Helps decide ARIMA's 'q' parameter (MA part).

    Parameters:
        df (DataFrame): Dataset
        column_name (str): Target column
        output_path (str): File path to save plot
    """

    print("[INFO] Plotting ACF...")

    save_plot(output_path)

    plt.figure(figsize=(10, 5))
    sm.graphics.tsa.plot_acf(df[column_name], lags=40)
    plt.title("Autocorrelation Function (ACF)")

    plt.savefig(output_path)
    plt.close()

    print(f"[INFO] ACF plot saved to {output_path}")


def plot_pacf(df, column_name, output_path="results/plots/pacf.png"):
    """
    Plots the Partial Autocorrelation Function (PACF).

    Why PACF?
    - Shows correlation of a series with lagged values while removing
      the influence of intermediate lags.
    - Helps decide ARIMA's 'p' parameter (AR part).

    Parameters:
        df (DataFrame): Dataset
        column_name (str): Exchange rate column
        output_path (str): Where to save plot
    """

    print("[INFO] Plotting PACF...")

    save_plot(output_path)

    plt.figure(figsize=(10, 5))
    sm.graphics.tsa.plot_pacf(df[column_name], lags=40, method="ywm")
    plt.title("Partial Autocorrelation Function (PACF)")

    plt.savefig(output_path)
    plt.close()

    print(f"[INFO] PACF plot saved to {output_path}")


