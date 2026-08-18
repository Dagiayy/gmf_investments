import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, jarque_bera

def calculate_max_drawdown(cum_returns_series):
    """
    Computes Maximum Drawdown (MDD) and drawdown series from a cumulative returns series.
    """
    peak = cum_returns_series.cummax()
    drawdown = (cum_returns_series - peak) / peak
    max_drawdown = drawdown.min()
    return abs(max_drawdown), drawdown

def calculate_sortino_ratio(returns_series, risk_free_rate=0.02, trading_days=252):
    """
    Calculates Annualized Sortino Ratio using downside deviation.
    """
    rf_daily = risk_free_rate / trading_days
    excess_returns = returns_series - rf_daily
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.nan
        
    downside_std = downside_returns.std() * np.sqrt(trading_days)
    annualized_return = returns_series.mean() * trading_days
    return (annualized_return - risk_free_rate) / downside_std

def calculate_calmar_ratio(returns_series, max_dd, risk_free_rate=0.02, trading_days=252):
    """
    Calculates Calmar Ratio = Annualized Return / Maximum Drawdown.
    """
    if max_dd == 0:
        return np.nan
    ann_return = returns_series.mean() * trading_days
    return (ann_return - risk_free_rate) / max_dd

def calculate_expected_shortfall(returns_series, alpha=0.95):
    """
    Computes Expected Shortfall / Conditional Value at Risk (CVaR).
    """
    var_threshold = np.percentile(returns_series.dropna(), (1 - alpha) * 100)
    cvar = returns_series[returns_series <= var_threshold].mean()
    return abs(cvar)

def calculate_asset_beta(asset_returns, market_returns):
    """
    Calculates Beta of asset relative to market index (SPY).
    """
    df = pd.DataFrame({'Asset': asset_returns, 'Market': market_returns}).dropna()
    cov = df.cov().iloc[0, 1]
    market_var = df['Market'].var()
    if market_var == 0:
        return np.nan
    return cov / market_var

def calculate_distribution_stats(returns_series):
    """
    Computes distribution higher-order moments and Jarque-Bera normality test.
    """
    clean_ret = returns_series.dropna()
    sk = skew(clean_ret)
    kt = kurtosis(clean_ret) # Excess kurtosis
    jb_stat, jb_pval = jarque_bera(clean_ret)
    
    return {
        "Skewness": sk,
        "Kurtosis": kt,
        "Jarque-Bera Stat": jb_stat,
        "Jarque-Bera p-value": jb_pval,
        "Is Normal (p>0.05)": jb_pval > 0.05
    }

def calculate_comprehensive_risk_metrics(returns_series, market_returns=None, risk_free_rate=0.02, trading_days=252):
    """
    Computes a comprehensive risk profile dictionary for a return series.
    """
    clean_ret = returns_series.dropna()
    mean_daily = clean_ret.mean()
    std_daily = clean_ret.std()
    
    ann_return = mean_daily * trading_days
    ann_vol = std_daily * np.sqrt(trading_days)
    
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    
    cum_returns = (1 + clean_ret).cumprod()
    max_dd, drawdown_series = calculate_max_drawdown(cum_returns)
    
    sortino = calculate_sortino_ratio(clean_ret, risk_free_rate=risk_free_rate, trading_days=trading_days)
    calmar = calculate_calmar_ratio(clean_ret, max_dd, risk_free_rate=risk_free_rate, trading_days=trading_days)
    
    var_95 = abs(np.percentile(clean_ret, 5))
    var_99 = abs(np.percentile(clean_ret, 1))
    
    cvar_95 = calculate_expected_shortfall(clean_ret, alpha=0.95)
    cvar_99 = calculate_expected_shortfall(clean_ret, alpha=0.99)
    
    beta = calculate_asset_beta(clean_ret, market_returns) if market_returns is not None else np.nan
    dist_stats = calculate_distribution_stats(clean_ret)
    
    return {
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_dd,
        "95% Value at Risk (VaR)": var_95,
        "99% Value at Risk (VaR)": var_99,
        "95% Expected Shortfall (CVaR)": cvar_95,
        "99% Expected Shortfall (CVaR)": cvar_99,
        "Beta to Benchmark": beta,
        "Skewness": dist_stats["Skewness"],
        "Kurtosis": dist_stats["Kurtosis"],
        "Jarque-Bera p-value": dist_stats["Jarque-Bera p-value"]
    }
