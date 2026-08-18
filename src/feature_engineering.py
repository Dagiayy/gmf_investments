import os
import numpy as np
import pandas as pd

def compute_rsi(prices, window=14):
    """Computes Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, 1e-6)
    return 100 - (100 / (1 + rs))

def compute_macd(prices, fast=12, slow=26, signal=9):
    """Computes Moving Average Convergence Divergence (MACD)."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def compute_bollinger_bands(prices, window=20, num_std=2):
    """Computes Bollinger Bands (Upper, Middle, Lower, %B)."""
    sma = prices.rolling(window=window).mean()
    rstd = prices.rolling(window=window).std()
    upper = sma + (rstd * num_std)
    lower = sma - (rstd * num_std)
    pct_b = (prices - lower) / (upper - lower).replace(0, 1e-6)
    return upper, sma, lower, pct_b

def extract_asset_features(df_asset, asset_name="Asset", output_dir="02_features"):
    """
    Extracts a comprehensive technical feature matrix for an asset price DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    price_col = 'Adj Close' if 'Adj Close' in df_asset.columns else 'Close'
    prices = df_asset[price_col]
    
    df_feat = pd.DataFrame(index=df_asset.index)
    df_feat['Price'] = prices
    df_feat['Log_Return'] = np.log(prices / prices.shift(1))
    df_feat['Simple_Return'] = prices.pct_change()
    
    # Lagged features
    df_feat['Lag1_Return'] = df_feat['Log_Return'].shift(1)
    df_feat['Lag2_Return'] = df_feat['Log_Return'].shift(2)
    df_feat['Lag5_Return'] = df_feat['Log_Return'].shift(5)
    
    # Moving Averages
    df_feat['SMA_20'] = prices.rolling(window=20).mean()
    df_feat['SMA_50'] = prices.rolling(window=50).mean()
    df_feat['SMA_200'] = prices.rolling(window=200).mean()
    df_feat['EMA_12'] = prices.ewm(span=12, adjust=False).mean()
    df_feat['EMA_26'] = prices.ewm(span=26, adjust=False).mean()
    
    # Technical Indicators
    df_feat['RSI_14'] = compute_rsi(prices, window=14)
    macd, signal, hist = compute_macd(prices)
    df_feat['MACD'] = macd
    df_feat['MACD_Signal'] = signal
    df_feat['MACD_Hist'] = hist
    
    upper, mid, lower, pct_b = compute_bollinger_bands(prices)
    df_feat['BB_Upper'] = upper
    df_feat['BB_Lower'] = lower
    df_feat['BB_PctB'] = pct_b
    
    # Volatility Indicators
    df_feat['Rolling_Vol_20D'] = df_feat['Log_Return'].rolling(window=20).std() * np.sqrt(252)
    
    df_feat = df_feat.dropna()
    out_path = os.path.join(output_dir, f"{asset_name.lower()}_features.csv")
    df_feat.to_csv(out_path)
    print(f"Extracted feature matrix for {asset_name.upper()} saved at {out_path}")
    return df_feat
