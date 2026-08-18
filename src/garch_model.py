import numpy as np
import pandas as pd
from scipy.optimize import minimize

class GARCHModel:
    """
    Generalized Autoregressive Conditional Heteroskedasticity (GARCH(1,1)) model.
    Models conditional variance: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    """
    def __init__(self, omega=None, alpha=None, beta=None):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.fitted = False

    def _log_likelihood(self, params, returns):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1.0:
            return 1e10
            
        n = len(returns)
        variance = np.zeros(n)
        variance[0] = np.var(returns)
        
        for t in range(1, n):
            variance[t] = omega + alpha * (returns[t-1] ** 2) + beta * variance[t-1]
            if variance[t] <= 0:
                variance[t] = 1e-6
                
        log_lik = -0.5 * np.sum(np.log(2 * np.pi * variance) + (returns ** 2) / variance)
        return -log_lik

    def fit(self, returns_series):
        """Fits GARCH(1,1) parameters via Maximum Likelihood Estimation."""
        clean_ret = returns_series.dropna().values
        uncond_var = np.var(clean_ret)
        
        init_params = [uncond_var * 0.05, 0.10, 0.85]
        bounds = [(1e-6, None), (0.001, 0.5), (0.5, 0.99)]
        
        res = minimize(self._log_likelihood, init_params, args=(clean_ret,), method='L-BFGS-B', bounds=bounds)
        
        self.omega, self.alpha, self.beta = res.x
        self.fitted = True
        
        # Calculate fitted conditional volatility series
        cond_vol = np.zeros(len(clean_ret))
        cond_vol[0] = np.sqrt(uncond_var)
        for t in range(1, len(clean_ret)):
            var_t = self.omega + self.alpha * (clean_ret[t-1] ** 2) + self.beta * (cond_vol[t-1] ** 2)
            cond_vol[t] = np.sqrt(max(var_t, 1e-6))
            
        return pd.Series(cond_vol, index=returns_series.dropna().index)

    def forecast_volatility(self, returns_series, horizon_days=252):
        """Forecasts dynamic conditional volatility over future horizon."""
        if not self.fitted:
            self.fit(returns_series)
            
        clean_ret = returns_series.dropna().values
        last_var = self.omega + self.alpha * (clean_ret[-1] ** 2) + self.beta * np.var(clean_ret)
        uncond_var = self.omega / (1.0 - self.alpha - self.beta) if (1 - self.alpha - self.beta) > 0 else np.var(clean_ret)
        
        vol_forecasts = []
        current_var = last_var
        
        for h in range(horizon_days):
            current_var = self.omega + (self.alpha + self.beta) * current_var
            vol_forecasts.append(np.sqrt(max(current_var, 1e-6)) * np.sqrt(252)) # Annualized
            
        return np.array(vol_forecasts)

def fit_asset_garch_volatility(df_returns):
    """
    Fits GARCH(1,1) model for each asset and returns DataFrame of conditional volatility series.
    """
    garch_vols = {}
    for col in df_returns.columns:
        model = GARCHModel()
        vol_series = model.fit(df_returns[col])
        garch_vols[col] = vol_series * np.sqrt(252) # Annualized
        
    return pd.DataFrame(garch_vols)
