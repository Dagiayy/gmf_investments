# src/task2_modeling.py
import os
import pandas as pd
import matplotlib.pyplot as plt

from models import (
    load_series, train_test_split_series,
    adf_test, find_best_arima_by_aic, arima_forecast, save_arima_model,
    train_lstm, compute_metrics
)

plt.style.use('seaborn-darkgrid')


def plot_results(dates, actual, preds, conf_int=None, title="Forecast", path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, preds, label='Forecast', linestyle='--')
    if conf_int is not None:
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]
        plt.fill_between(dates, lower, upper, color='lightgrey', alpha=0.4, label='95% CI')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path)
        print("Saved plot:", path)
    plt.close()


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    series = load_series("data/processed/tsla_processed.csv")
    train, test = train_test_split_series(series, train_end='2023-12-31')
    steps = len(test)
    print(f"Train length: {len(train)}  Test length: {len(test)}")

    # ----- Check stationarity (informational) -----
    adf_res = adf_test(series)
    print("ADF on full series p-value:", adf_res["pvalue"])

    # ---------------- ARIMA ----------------
    print("\n--- ARIMA grid-search (AIC) ---")
    # First try a modest grid
    arima_fit, arima_order = find_best_arima_by_aic(train, p_range=(0, 4), d_range=(0, 2), q_range=(0, 4))
    if arima_fit is None:
        print("No ARIMA found in initial grid. Trying fallback with d=1 and expanded p/q.")
        arima_fit, arima_order = find_best_arima_by_aic(train, p_range=(0, 6), d_range=(1, 1), q_range=(0, 6))

    if arima_fit is None:
        raise RuntimeError("ARIMA search failed. Consider manual inspection, differencing the series, or expanding ranges.")

    print("Best ARIMA order:", arima_order)
    save_arima_model(arima_fit, "models/arima")
    arima_preds, arima_ci = arima_forecast(arima_fit, steps=steps)
    # align index
    arima_preds.index = test.index
    arima_ci.index = test.index

    arima_metrics = compute_metrics(test.values, arima_preds.values)
    print("ARIMA metrics:", arima_metrics)
    plot_results(test.index, test.values, arima_preds.values, conf_int=arima_ci,
                 title="TSLA ARIMA Forecast vs Actual", path="output/tsla_arima_forecast.png")

    # ---------------- LSTM ----------------
    print("\n--- LSTM training & forecasting ---")
    lstm_preds, history, scaler = train_lstm(train, test, window=60, units=50, epochs=50,
                                             batch_size=32, patience=7, save_path="models/lstm")
    lstm_series = pd.Series(lstm_preds, index=test.index)
    lstm_metrics = compute_metrics(test.values, lstm_series.values)
    print("LSTM metrics:", lstm_metrics)
    plot_results(test.index, test.values, lstm_series.values,
                 title="TSLA LSTM Forecast vs Actual", path="output/tsla_lstm_forecast.png")

    # Save metrics table
    metrics_df = pd.DataFrame([arima_metrics, lstm_metrics], index=['ARIMA', 'LSTM'])
    metrics_df.to_csv("output/model_metrics.csv")
    print("Saved metrics to output/model_metrics.csv")
    print(metrics_df)


if __name__ == "__main__":
    main()
