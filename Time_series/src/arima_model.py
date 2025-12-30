# arima_model.py
# ---------------------------------------------------------------
# This file builds, trains, and forecasts using the ARIMA model.
#
# ARIMA Model Components:
#   p → Auto-Regressive term
#   d → Differencing term (to make the series stationary)
#   q → Moving Average term
#
# We also generate future predictions and save the model.
# ---------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")   # Hide ARIMA warnings

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import joblib                          # Used to save and load the model


def train_arima_model(df, column_name, order=(1, 1, 1), model_path="models/arima_model.pkl"):
    """
    Trains an ARIMA model.

    Parameters:
        df (DataFrame): Cleaned time-series dataset
        column_name (str): The column to model (exchange rate)
        order (tuple): (p, d, q) parameters for ARIMA
        model_path (str): File path to save the model

    Returns:
        model_fit: Trained ARIMA model
    """

    print(f"[INFO] Training ARIMA model with order={order}...")

    # Extract the series to model
    series = df[column_name]

    # Build the ARIMA model object
    model = sm.tsa.ARIMA(series, order=order)

    # Fit the model to the data
    model_fit = model.fit()

    print("[INFO] ARIMA model training completed.")
    print(model_fit.summary())

    # Save the trained model for later use
    joblib.dump(model_fit, model_path)
    print(f"[INFO] ARIMA model saved to {model_path}")

    return model_fit


def forecast_arima(model_fit, steps, output_csv="results/forecasts/arima_forecast_values.csv"):
    """
    Forecasts future values using a trained ARIMA model.

    Parameters:
        model_fit: The trained ARIMA model
        steps (int): Number of future time steps to predict
        output_csv (str): File path to save forecast results

    Returns:
        forecast_df (DataFrame): Forecasted values
    """

    print(f"[INFO] Forecasting next {steps} steps using ARIMA...")

    # Generate forecast values
    forecast = model_fit.forecast(steps=steps)

    # Convert to DataFrame for easier handling
    forecast_df = pd.DataFrame({
        "Forecast": forecast
    })

    # Save forecast to CSV file
    forecast_df.to_csv(output_csv, index=True)
    print(f"[INFO] ARIMA forecast saved to {output_csv}")

    return forecast_df


def plot_arima_forecast(df, forecast_df, column_name, output_path="results/plots/arima_forecast.png"):
    """
    Visualizes the ARIMA forecast vs actual historical data.

    Parameters:
        df (DataFrame): Actual historical dataset
        forecast_df (DataFrame): DataFrame containing forecasted values
        column_name (str): Name of the actual exchange rate column
        output_path (str): Path to save plot image
    """

    print("[INFO] Plotting ARIMA forecast...")

    plt.figure(figsize=(10, 5))

    # Plot actual data
    plt.plot(df[column_name], label="Actual Data")

    # Plot forecast
    plt.plot(forecast_df["Forecast"], label="Forecast", linestyle="--")

    plt.title("ARIMA Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate")
    plt.legend()

    plt.savefig(output_path)
    plt.close()

    print(f"[INFO] ARIMA forecast plot saved to {output_path}")


