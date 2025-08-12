# src/models.py
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# statsmodels ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Keras for LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")


# -------------------------
# Data utilities
# -------------------------
def load_series(processed_csv):
    """
    Load processed CSV and return a pandas Series of price (Adj Close if available else Close).
    """
    df = pd.read_csv(processed_csv, index_col=0, parse_dates=True)
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    series = df[price_col].astype(float).rename('Price')
    return series


def train_test_split_series(series, train_end='2023-12-31'):
    """
    Chronological split. train_end is inclusive.
    """
    train = series.loc[:train_end]
    test = series.loc[train_end:]
    if len(test) > 0 and test.index[0] == train.index[-1]:
        test = test.iloc[1:]
    return train, test


# -------------------------
# Stationarity helpers
# -------------------------
def adf_test(series):
    """
    Runs Augmented Dickey-Fuller test. Returns dict with statistic, pvalue, crit vals.
    """
    series = series.dropna()
    res = adfuller(series)
    return {
        "adf_stat": res[0],
        "pvalue": res[1],
        "nobs": res[2],
        "crit_vals": res[4]
    }


def difference_series(series, periods=1):
    return series.diff(periods=periods).dropna()


# -------------------------
# Metrics
# -------------------------
def mape(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    denom = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape(y_true, y_pred)}


# -------------------------
# ARIMA grid-search by AIC
# -------------------------
def find_best_arima_by_aic(series, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3), maxiter=50):
    """
    Grid search across (p,d,q) by AIC using statsmodels' ARIMA. Returns fitted model and order.
    Use modest ranges to avoid long runtime. If nothing fits, returns (None, None).
    """
    best_aic = np.inf
    best_order = None
    best_model = None

    y = series.astype(float)

    tried = 0
    for p in range(p_range[0], p_range[1] + 1):
        for d in range(d_range[0], d_range[1] + 1):
            for q in range(q_range[0], q_range[1] + 1):
                tried += 1
                try:
                    model = ARIMA(y, order=(p, d, q))
                    fitted = model.fit(method_kwargs={"maxiter": maxiter})
                    aic = getattr(fitted, "aic", np.inf)
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = fitted
                except Exception:
                    # fit failed for this combination; skip
                    continue

    return best_model, best_order


def arima_forecast(model_fit, steps):
    """
    Returns forecast mean Series and conf_int DataFrame (both indexed by integer positions;
    caller should align with test.index).
    """
    fc = model_fit.get_forecast(steps=steps)
    mean_forecast = fc.predicted_mean
    conf_int = fc.conf_int(alpha=0.05)
    return mean_forecast, conf_int


# -------------------------
# LSTM helpers
# -------------------------
def create_sequences(values, window):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])
        y.append(values[i + window])
    return np.array(X), np.array(y)


def build_lstm(input_shape, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm(train_series, test_series, window=60, units=50, epochs=50, batch_size=32, patience=7,
               save_path=None, use_recursive_forecast=True):
    """
    Train an LSTM and forecast test_series length using recursive forecasting by default.
    Returns predictions array (floats), training history, and scaler.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    scaler = MinMaxScaler()
    train_vals = train_series.values.reshape(-1, 1)
    test_vals = test_series.values.reshape(-1, 1)

    scaler.fit(train_vals)
    train_scaled = scaler.transform(train_vals).flatten()
    test_scaled = scaler.transform(test_vals).flatten()

    X_train, y_train = create_sequences(train_scaled, window)
    if X_train.size == 0:
        raise ValueError("Window too large for training data. Reduce window size.")

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = build_lstm((X_train.shape[1], X_train.shape[2]), units=units, dropout=0.2)
    es = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[es])

    # Forecast on test set length
    preds_scaled = []
    last_window = list(train_scaled[-window:])

    for i in range(len(test_scaled)):
        x = np.array(last_window[-window:]).reshape(1, window, 1)
        yhat = model.predict(x, verbose=0)[0][0]
        if use_recursive_forecast:
            last_window.append(yhat)
        else:
            last_window.append(test_scaled[i])
        preds_scaled.append(yhat)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds_scaled).flatten()

    if save_path:
        model.save(os.path.join(save_path, "lstm_model.h5"))
        joblib.dump(scaler, os.path.join(save_path, "lstm_scaler.pkl"))

    return preds_inv, history, scaler


# -------------------------
# Save / load helpers
# -------------------------
def save_arima_model(model_fit, path):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model_fit, os.path.join(path, "arima_model.pkl"))


def load_arima_model(path):
    return joblib.load(os.path.join(path, "arima_model.pkl"))
