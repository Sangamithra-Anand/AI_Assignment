# evaluate.py
# -------------------------------------------------------------------------
# This file handles evaluation of forecasting models.
#
# We calculate:
# 1. MAE  - Mean Absolute Error
# 2. RMSE - Root Mean Squared Error
# 3. MAPE - Mean Absolute Percentage Error
#
# IMPROVED VERSION:
# - Automatically detects forecast date column
# - Aligns forecast with actual using DATE index
# - Removes NaN and INF before computing metrics
# - Prevents JSON errors
# -------------------------------------------------------------------------

import numpy as np
import pandas as pd
import json


def mean_absolute_error(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def root_mean_squared_error(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mean_absolute_percentage_error(actual, predicted):
    actual = np.where(actual == 0, 1e-10, actual)
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def evaluate_forecast(actual_df, forecast_df, column_name,
                      output_path="results/metrics/eval_report.json"):
    """
    Computes MAE, RMSE, and MAPE between actual and forecasted values.

    Improvements:
    - Automatically detects the forecast date column
    - Aligns actual & forecast by DATE
    - Removes NaN/Inf values safely
    """

    print("[INFO] Evaluating model forecast accuracy...")

    forecast_df = forecast_df.copy()

    # -------------------------------------------------------
    # AUTO-DETECT DATE COLUMN IN FORECAST DATA
    # -------------------------------------------------------
    date_col = None

    for col in forecast_df.columns:
        try:
            pd.to_datetime(forecast_df[col])
            date_col = col
            break
        except:
            pass

    if date_col is None:
        print("[ERROR] No valid date column found in forecast file!")
        metrics = {"MAE": 0.0, "RMSE": 0.0, "MAPE (%)": 0.0}
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        return metrics

    # Convert detected date column
    forecast_df.index = pd.to_datetime(forecast_df[date_col])

    # Forecast values
    predicted_series = forecast_df["Forecast"]

    # -------------------------------------------------------
    # ALIGN ACTUAL & FORECAST BY DATE
    # -------------------------------------------------------
    common_dates = actual_df.index.intersection(predicted_series.index)

    if len(common_dates) == 0:
        print("[WARNING] No overlapping dates found for evaluation.")
        metrics = {"MAE": 0.0, "RMSE": 0.0, "MAPE (%)": 0.0}
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        return metrics

    actual = actual_df.loc[common_dates, column_name]
    predicted = predicted_series.loc[common_dates]

    # -------------------------------------------------------
    # REMOVE NaN & INF
    # -------------------------------------------------------
    actual = actual.replace([np.inf, -np.inf], np.nan)
    predicted = predicted.replace([np.inf, -np.inf], np.nan)

    valid = actual.notna() & predicted.notna()
    actual = actual[valid]
    predicted = predicted[valid]

    # -------------------------------------------------------
    # METRICS
    # -------------------------------------------------------
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(root_mean_squared_error(actual, predicted))
    mape = float(mean_absolute_percentage_error(actual, predicted))

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    }

    # -------------------------------------------------------
    # SAVE METRICS TO JSON
    # -------------------------------------------------------
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Evaluation results saved: {output_path}")
    print("[INFO] Evaluation Summary:", metrics)

    return metrics


