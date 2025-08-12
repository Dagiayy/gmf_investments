import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(file_path):
    """Load CSV and ensure datetime index."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def basic_stats(df):
    """Return basic statistics for numeric columns."""
    return df.describe()

def check_missing(df):
    """Check for missing values per column."""
    return df.isnull().sum()

def fill_missing(df, method='ffill'):
    """Fill missing values with forward fill or interpolation."""
    if method == 'ffill':
        return df.ffill()
    elif method == 'bfill':
        return df.bfill()
    elif method == 'interpolate':
        return df.interpolate(method='linear')
    else:
        raise ValueError("Invalid method. Use 'ffill', 'bfill', or 'interpolate'.")

def scale_data(df, cols, method='minmax'):
    """Scale data using MinMax or Standard scaling."""
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    scaled_values = scaler.fit_transform(df[cols])
    scaled_df = pd.DataFrame(scaled_values, columns=cols, index=df.index)
    return scaled_df, scaler
