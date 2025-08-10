import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

def load_processed_data(file_path):
    """Load processed CSV with Date index."""
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

def plot_closing_price(df, output_folder, asset):
    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.title(f'{asset.upper()} Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(output_folder, f'{asset}_closing_price.png')
    plt.savefig(path)
    print(f"Saved closing price plot to {path}")
    plt.close()

def plot_daily_pct_change(df, output_folder, asset):
    df['Daily_Pct_Change'] = df['Close'].pct_change() * 100
    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['Daily_Pct_Change'], label='Daily % Change', color='orange')
    plt.title(f'{asset.upper()} Daily Percentage Change')
    plt.xlabel('Date')
    plt.ylabel('Daily % Change')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(output_folder, f'{asset}_daily_pct_change.png')
    plt.savefig(path)
    print(f"Saved daily % change plot to {path}")
    plt.close()

def volatility_analysis(df, output_folder, asset, window=30):
    df['Rolling_Mean'] = df['Daily_Pct_Change'].rolling(window=window).mean()
    df['Rolling_Std'] = df['Daily_Pct_Change'].rolling(window=window).std()

    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['Rolling_Mean'], label=f'{window}-day Rolling Mean')
    plt.plot(df.index, df['Rolling_Std'], label=f'{window}-day Rolling Std Dev')
    plt.title(f'{asset.upper()} {window}-Day Rolling Mean and Std Dev of Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Percent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(output_folder, f'{asset}_rolling_volatility.png')
    plt.savefig(path)
    print(f"Saved rolling volatility plot to {path}")
    plt.close()

def detect_outliers(df):
    mean = df['Daily_Pct_Change'].mean()
    std = df['Daily_Pct_Change'].std()
    cutoff = 3 * std
    outliers = df[(df['Daily_Pct_Change'] > mean + cutoff) | (df['Daily_Pct_Change'] < mean - cutoff)]
    print(f"Detected {len(outliers)} outliers with daily % change beyond ±3 std dev")
    return outliers

def adf_test(series, series_name):
    print(f"\nAugmented Dickey-Fuller Test for {series_name}:")
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    if result[1] < 0.05:
        print(f"Conclusion: {series_name} is stationary (reject H0)")
    else:
        print(f"Conclusion: {series_name} is non-stationary (fail to reject H0)")

def calculate_risk_metrics(df):
    daily_returns = df['Close'].pct_change().dropna()
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    VaR_95 = mean_return - 1.65 * std_return
    sharpe_ratio = mean_return / std_return * np.sqrt(252)
    print(f"\nRisk Metrics:")
    print(f"Mean daily return: {mean_return:.6f}")
    print(f"Daily return std dev: {std_return:.6f}")
    print(f"95% Value at Risk (VaR): {VaR_95:.6f}")
    print(f"Annualized Sharpe Ratio (Rf=0): {sharpe_ratio:.4f}")

def main():
    assets = ['tsla', 'bnd', 'spy']
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for asset in assets:
        processed_path = f"data/processed/{asset}_processed.csv"
        print(f"\n=== Analyzing {asset.upper()} ===")
        df = load_processed_data(processed_path)

        plot_closing_price(df, output_folder, asset)
        plot_daily_pct_change(df, output_folder, asset)
        volatility_analysis(df, output_folder, asset)

        outliers = detect_outliers(df)
        if not outliers.empty:
            print(outliers[['Close', 'Daily_Pct_Change']])
        else:
            print("No significant outliers detected.")

        adf_test(df['Close'], f"{asset.upper()} Closing Price")
        adf_test(df['Daily_Pct_Change'], f"{asset.upper()} Daily Percentage Change")

        calculate_risk_metrics(df)

if __name__ == "__main__":
    main()
