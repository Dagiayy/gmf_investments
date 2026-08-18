import pandas as pd
import numpy as np

def validate_price_df(df, asset_name="Asset"):
    """
    Validates market price DataFrame for required schema, data types, 
    missing values, non-positive values, and date continuity.
    """
    if df is None or df.empty:
        raise ValueError(f"Data contract violation: Dataframe for {asset_name} is empty or None.")
        
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Data contract violation: {asset_name} index must be DatetimeIndex.")
        
    if df.index.duplicated().any():
        num_dups = df.index.duplicated().sum()
        raise ValueError(f"Data contract violation: {asset_name} contains {num_dups} duplicate date index entries.")
        
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    if price_col not in df.columns:
        raise KeyError(f"Data contract violation: Neither 'Adj Close' nor 'Close' column found for {asset_name}.")
        
    prices = df[price_col]
    if (prices <= 0).any():
        raise ValueError(f"Data contract violation: Non-positive price values detected in {asset_name}.")
        
    if prices.isna().any():
        raise ValueError(f"Data contract violation: NaNs detected in price series for {asset_name}.")

    return True

def validate_aligned_returns(returns_df):
    """
    Validates aligned returns DataFrame across multiple assets.
    Checks positive semi-definiteness of covariance matrix and NaN absence.
    """
    if returns_df is None or returns_df.empty:
        raise ValueError("Data contract violation: Returns DataFrame is empty or None.")
        
    if returns_df.isna().any().any():
        raise ValueError("Data contract violation: NaNs present in aligned returns DataFrame.")
        
    cov_matrix = returns_df.cov().values
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    if np.any(eigenvalues < -1e-8):
        raise ValueError("Data contract violation: Covariance matrix is not positive semi-definite.")
        
    return True

def validate_portfolio_weights(weights):
    """
    Validates portfolio allocation weights vector.
    """
    if isinstance(weights, dict):
        w_vals = np.array(list(weights.values()))
    else:
        w_vals = np.array(weights)
        
    if np.any(w_vals < -1e-6):
        raise ValueError(f"Data contract violation: Negative weight detected in long-only portfolio: {weights}")
        
    if not np.isclose(np.sum(w_vals), 1.0, atol=1e-4):
        raise ValueError(f"Data contract violation: Portfolio weights do not sum to 1.0 (Sum = {np.sum(w_vals):.6f})")
        
    return True
