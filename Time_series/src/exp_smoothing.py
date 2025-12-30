# exp_smoothing.py
# -------------------------------------------------------------------
# This file builds, trains, and forecasts using Exponential Smoothing
# models including:
#
# 1. Simple Exponential Smoothing
# 2. Holt's Linear Trend method
# 3. Holt-Winters Seasonal method (Additive)
#
# These models are alternatives to ARIMA and often perform better
# for trend/seasonal data.
# -------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    ExponentialSmoothing,
    Holt
)
import joblib   # Used to save trained models


def train_exp_smoothing(df, column_name, model_type="holt", seasonal_periods=None,
                        model_path="models/exp_smoothing.pkl"):
    """
    Trains an Exponential Smoothing model.

    Parameters:
        df (DataFrame): Cleaned dataset
        column_name (str): Name of the exchange rate column
        model_type (str): "simple", "holt", or "holt-winters"
        seasonal_periods (int): Seasonal cycle length (if required)
        model_path (str): Where to save the trained model

    Returns:
        model_fit: Trained model
    """

    series = df[column_name]

    print(f"[INFO] Training Exponential Smoothing model: {model_type}")

    # ------------------------------
    # 1. SIMPLE EXPONENTIAL SMOOTHING
    # ------------------------------
    if model_type == "simple":
        model = SimpleExpSmoothing(series)
        model_fit = model.fit()

    # ------------------------------
    # 2. HOLT'S LINEAR TREND METHOD
    # ------------------------------
    elif model_type == "holt":
        model = Holt(series)
        model_fit = model.fit()

    # ------------------------------
    # 3. HOLT-WINTERS (Seasonal)
    # ------------------------------
    elif model_type == "holt-winters":
        if seasonal_periods is None:
            raise ValueError("[ERROR] seasonal_periods is required for Holt-Winters method.")

        model = ExponentialSmoothing(
            series,
            trend="add",                 # Additive trend component
            seasonal="add",              # Additive seasonal pattern
            seasonal_periods=seasonal_periods
        )
        model_fit = model.fit()

    else:
        raise ValueError("[ERROR] model_type must be: simple / holt / holt-winters")

    # Save trained model
    joblib.dump(model_fit, model_path)
    print(f"[INFO] Exponential Smoothing model saved to {model_path}")

    return model_fit


def forecast_exp_smoothing(model_fit, steps, output_csv="results/forecasts/exp_smoothing_values.csv"):
    """
    Forecasts future values using an Exponential Smoothing model.

    Parameters:
        model_fit: Trained exponential smoothing model
        steps (int): Number of future values to predict
        output_csv (str): Where to save the forecast

    Returns:
        forecast_df (DataFrame): Forecasted values
    """

    print(f"[INFO] Forecasting next {steps} steps with Exponential Smoothing...")

    forecast = model_fit.forecast(steps)

    # Convert to DataFrame for consistency
    forecast_df = pd.DataFrame({"Forecast": forecast})

    forecast_df.to_csv(output_csv)
    print(f"[INFO] Exponential Smoothing forecast saved to {output_csv}")

    return forecast_df


def plot_exp_smoothing(df, forecast_df, column_name, output_path="results/plots/exp_smoothing_forecast.png"):
    """
    Plots actual vs forecasted values.

    Parameters:
        df (DataFrame): Historical dataset
        forecast_df (DataFrame): Future predicted values
        column_name (str): Name of actual exchange rate column
        output_path (str): File path to save plot
    """

    print("[INFO] Plotting Exponential Smoothing forecast...")

    plt.figure(figsize=(10, 5))

    # Plot actual values
    plt.plot(df[column_name], label="Actual Data")

    # Plot forecast
    plt.plot(forecast_df["Forecast"], label="Forecast", linestyle="--")

    plt.title("Exponential Smoothing Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate")
    plt.legend()

    plt.savefig(output_path)
    plt.close()

    print(f"[INFO] Exponential Smoothing forecast plot saved to {output_path}")



