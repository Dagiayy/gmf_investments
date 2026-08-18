import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.data_contracts import validate_portfolio_weights

def calculate_portfolio_performance(weights, exp_returns, cov_matrix):
    """Calculates expected return and annualized volatility for a given weight vector."""
    weights = np.array(weights)
    p_return = np.sum(weights * exp_returns)
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return p_return, p_volatility

def estimate_shrinkage_covariance(returns_df, trading_days=252):
    """
    Estimates annualized covariance matrix using Ledoit-Wolf shrinkage 
    to reduce sample estimation noise in portfolio optimization.
    """
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        shrunk_cov = lw.fit(returns_df.dropna()).covariance_ * trading_days
        return pd.DataFrame(shrunk_cov, index=returns_df.columns, columns=returns_df.columns)
    except Exception:
        return returns_df.cov() * trading_days

def estimate_capm_expected_returns(returns_df, market_col='SPY', risk_free_rate=0.02, trading_days=252):
    """
    Estimates expected returns using Capital Asset Pricing Model (CAPM):
    E(R_i) = R_f + Beta_i * (E(R_m) - R_f)
    """
    mkt_returns = returns_df[market_col]
    mkt_ann_ret = mkt_returns.mean() * trading_days
    equity_risk_premium = mkt_ann_ret - risk_free_rate
    
    capm_rets = {}
    mkt_var = mkt_returns.var()
    for col in returns_df.columns:
        if col == market_col:
            capm_rets[col] = mkt_ann_ret
        else:
            beta = returns_df[[col, market_col]].cov().iloc[0, 1] / mkt_var if mkt_var > 0 else 1.0
            capm_rets[col] = risk_free_rate + beta * equity_risk_premium
            
    return pd.Series(capm_rets)

def optimize_max_sharpe(exp_returns, cov_matrix, risk_free_rate=0.02):
    """
    Computes Maximum Sharpe Ratio Portfolio weights subject to sum(w)=1 and w_i >= 0.
    """
    num_assets = len(exp_returns)
    
    def neg_sharpe(w):
        p_ret, p_vol = calculate_portfolio_performance(w, exp_returns, cov_matrix)
        if p_vol == 0:
            return 1e6
        return -((p_ret - risk_free_rate) / p_vol)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = np.ones(num_assets) / num_assets
    
    res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = res.x / np.sum(res.x)
    validate_portfolio_weights(weights)
    
    p_ret, p_vol = calculate_portfolio_performance(weights, exp_returns, cov_matrix)
    sharpe = (p_ret - risk_free_rate) / p_vol
    
    return {
        "weights": dict(zip(exp_returns.index, weights)),
        "expected_return": p_ret,
        "volatility": p_vol,
        "sharpe_ratio": sharpe
    }

def optimize_min_volatility(cov_matrix, asset_names=None):
    """
    Computes Minimum Volatility Portfolio weights.
    """
    num_assets = cov_matrix.shape[0]
    if asset_names is None:
        asset_names = cov_matrix.index if hasattr(cov_matrix, 'index') else [f"Asset_{i}" for i in range(num_assets)]
        
    def portfolio_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = np.ones(num_assets) / num_assets
    
    res = minimize(portfolio_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = res.x / np.sum(res.x)
    validate_portfolio_weights(weights)
    
    p_vol = portfolio_vol(weights)
    
    return {
        "weights": dict(zip(asset_names, weights)),
        "volatility": p_vol
    }

def optimize_risk_parity(cov_matrix, asset_names=None):
    """
    Risk Parity (Equal Risk Contribution - ERC) Portfolio Optimization.
    Finds weights where each asset contributes equally to overall portfolio risk.
    """
    num_assets = cov_matrix.shape[0]
    if asset_names is None:
        asset_names = cov_matrix.index if hasattr(cov_matrix, 'index') else [f"Asset_{i}" for i in range(num_assets)]
        
    def risk_budget_objective(w):
        w = np.array(w).reshape(-1, 1)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))[0, 0]
        marginal_contrib = np.dot(cov_matrix, w) / p_vol
        risk_contrib = w * marginal_contrib
        
        target_risk_contrib = p_vol / num_assets
        return np.sum(np.square(risk_contrib - target_risk_contrib))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.001, 1.0) for _ in range(num_assets))
    init_guess = np.ones(num_assets) / num_assets
    
    res = minimize(risk_budget_objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = res.x / np.sum(res.x)
    validate_portfolio_weights(weights)
    
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    return {
        "weights": dict(zip(asset_names, weights)),
        "volatility": p_vol
    }

def optimize_black_litterman(market_caps, cov_matrix, views_dict, tau=0.05, risk_free_rate=0.02):
    """
    Black-Litterman Portfolio Optimization model.
    Combines market equilibrium prior returns with investor forecast views.
    """
    asset_names = cov_matrix.index if hasattr(cov_matrix, 'index') else list(market_caps.keys())
    num_assets = len(asset_names)
    
    total_cap = sum(market_caps.values())
    w_market = np.array([market_caps[a] / total_cap for a in asset_names])
    
    delta = 2.5
    pi = delta * np.dot(cov_matrix, w_market)
    
    num_views = len(views_dict)
    if num_views == 0:
        posterior_returns = pd.Series(pi, index=asset_names)
    else:
        P = np.zeros((num_views, num_assets))
        Q = np.zeros(num_views)
        
        for idx, (asset, view_ret) in enumerate(views_dict.items()):
            asset_idx = list(asset_names).index(asset)
            P[idx, asset_idx] = 1.0
            Q[idx] = view_ret
            
        tau_cov = tau * cov_matrix.values
        omega = np.diag(np.diag(np.dot(P, np.dot(tau_cov, P.T))))
        
        inv_tau_cov = np.linalg.inv(tau_cov)
        inv_omega = np.linalg.inv(omega)
        
        term1 = np.linalg.inv(inv_tau_cov + np.dot(P.T, np.dot(inv_omega, P)))
        term2 = np.dot(inv_tau_cov, pi) + np.dot(P.T, np.dot(inv_omega, Q))
        
        posterior_returns = pd.Series(np.dot(term1, term2), index=asset_names)
        
    opt = optimize_max_sharpe(posterior_returns, cov_matrix, risk_free_rate=risk_free_rate)
    opt["bl_expected_returns"] = posterior_returns.to_dict()
    return opt

def run_sensitivity_analysis(exp_returns, cov_matrix, target_asset='TSLA', return_shifts=[-0.25, -0.10, 0.0, 0.10, 0.25]):
    """
    Evaluates sensitivity of optimal portfolio weights when expected return of an asset shifts.
    """
    base_ret = exp_returns.copy()
    sensitivity_results = []
    
    for shift in return_shifts:
        mod_ret = base_ret.copy()
        mod_ret[target_asset] = base_ret[target_asset] * (1.0 + shift)
        opt = optimize_max_sharpe(mod_ret, cov_matrix)
        row = {"Return Shift (%)": f"{shift*100:+.0f}%", f"{target_asset} Exp Ret": f"{mod_ret[target_asset]:.2%}"}
        for a, w in opt["weights"].items():
            row[f"{a} Weight"] = f"{w:.2%}"
        row["Expected Return"] = f"{opt['expected_return']:.2%}"
        row["Volatility"] = f"{opt['volatility']:.2%}"
        row["Sharpe Ratio"] = f"{opt['sharpe_ratio']:.3f}"
        sensitivity_results.append(row)
        
    return pd.DataFrame(sensitivity_results)
