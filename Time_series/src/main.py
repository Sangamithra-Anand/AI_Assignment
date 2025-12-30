# main.py
# ---------------------------------------------------------------------------
# This is the MAIN CONTROLLER of your entire project.
#
# When you run:   python src/main.py
#
# It will:
#   1. Auto-create all required folders
#   2. Load dataset
#   3. Preprocess dataset
#   4. Visualize Time Series, ACF, PACF
#   5. Train ARIMA model + Forecast
#   6. Train Exponential Smoothing + Forecast
#   7. Evaluate both models
#
# Everything is explained clearly inside the code.
# ---------------------------------------------------------------------------

from utils import create_directories
from load_data import load_exchange_rate
from preprocess import preprocess_data
from visualize import plot_time_series, plot_acf, plot_pacf
from arima_model import train_arima_model, forecast_arima, plot_arima_forecast
from exp_smoothing import train_exp_smoothing, forecast_exp_smoothing, plot_exp_smoothing
from evaluate import evaluate_forecast


def main():

    print("="*70)
    print("          TIME SERIES FORECASTING PROJECT STARTED")
    print("="*70)

    # -------------------------------------------------------
    # 1. CREATE REQUIRED FOLDERS AUTOMATICALLY
    # -------------------------------------------------------
    create_directories()

    # -------------------------------------------------------
    # 2. LOAD DATASET
    # -------------------------------------------------------
    df = load_exchange_rate("data/exchange_rate.csv")

    # Only one column exists: exchange rate (USD → AUD)
    column_name = "Ex_rate"
    # -------------------------------------------------------
    # 3. PREPROCESS DATA (remove missing values + anomalies)
    # -------------------------------------------------------
    df = preprocess_data(df, column_name)

    # -------------------------------------------------------
    # 4. VISUALIZATIONS (for understanding & ARIMA tuning)
    # -------------------------------------------------------
    plot_time_series(df, column_name)
    plot_acf(df, column_name)
    plot_pacf(df, column_name)

    # -------------------------------------------------------
    # 5. ARIMA MODEL
    # -------------------------------------------------------
    print("\n=== TRAINING ARIMA MODEL ===")
    arima_model = train_arima_model(df, column_name, order=(1,1,1))

    print("\n=== FORECASTING WITH ARIMA ===")
    arima_forecast_df = forecast_arima(arima_model, steps=30)

    plot_arima_forecast(df, arima_forecast_df, column_name)

    # -------------------------------------------------------
    # 6. EXPONENTIAL SMOOTHING MODEL
    # -------------------------------------------------------
    print("\n=== TRAINING EXPONENTIAL SMOOTHING MODEL (Holt Method) ===")
    exp_model = train_exp_smoothing(df, column_name, model_type="holt")

    print("\n=== FORECASTING WITH EXPONENTIAL SMOOTHING ===")
    exp_forecast_df = forecast_exp_smoothing(exp_model, steps=30)

    plot_exp_smoothing(df, exp_forecast_df, column_name)

    # -------------------------------------------------------
    # 7. EVALUATION
    # -------------------------------------------------------
    print("\n=== EVALUATING ARIMA FORECAST ===")
    evaluate_forecast(df, arima_forecast_df, column_name,
                      output_path="results/metrics/arima_eval.json")

    print("\n=== EVALUATING EXPONENTIAL SMOOTHING FORECAST ===")
    evaluate_forecast(df, exp_forecast_df, column_name,
                      output_path="results/metrics/exp_eval.json")

    print("\n" + "="*70)
    print("        PROJECT COMPLETED SUCCESSFULLY!")
    print("        All results saved in the /results/ folder.")
    print("="*70)


# -------------------------------------------------------
# Standard Python entry point
# -------------------------------------------------------
if __name__ == "__main__":
    main()



