import numpy as np
import pandas as pd
from src.models import compute_metrics, naive_last_value_forecast, moving_average_forecast

class WalkForwardValidator:
    """
    Implements rolling-window / expanding-window time series cross-validation 
    to evaluate model stability across multiple historical regimes without data leakage.
    """
    def __init__(self, initial_train_size=1260, horizon=126, step_size=63):
        """
        initial_train_size: Number of business days for initial training (~5 years)
        horizon: Out-of-sample forecast window (~6 months)
        step_size: Step forward between folds (~3 months)
        """
        self.initial_train_size = initial_train_size
        self.horizon = horizon
        self.step_size = step_size

    def split(self, series):
        """
        Generates (train_idx, test_idx) slices for expanding window evaluation.
        """
        n = len(series)
        current_end = self.initial_train_size
        
        while current_end + self.horizon <= n:
            train_series = series.iloc[:current_end]
            test_series = series.iloc[current_end : current_end + self.horizon]
            yield train_series, test_series
            current_end += self.step_size

    def evaluate_baselines(self, series):
        """
        Runs walk-forward cross validation on naive baselines across all folds.
        Returns summary dictionary of mean metrics across folds.
        """
        naive_metrics = []
        ma_metrics = []
        
        for train, test in self.split(series):
            steps = len(test)
            
            # Naive Last-Value
            p_naive = naive_last_value_forecast(train, steps)
            m_naive = compute_metrics(test.values, p_naive)
            naive_metrics.append(m_naive)
            
            # Moving Average
            p_ma = moving_average_forecast(train, steps, window=30)
            m_ma = compute_metrics(test.values, p_ma)
            ma_metrics.append(m_ma)
            
        df_naive = pd.DataFrame(naive_metrics).mean()
        df_ma = pd.DataFrame(ma_metrics).mean()
        
        return {
            "Naive_Last_Value": df_naive.to_dict(),
            "Moving_Average_30D": df_ma.to_dict(),
            "Num_Folds": len(naive_metrics)
        }
