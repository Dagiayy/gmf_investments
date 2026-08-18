import numpy as np
import pandas as pd
from src.risk import calculate_comprehensive_risk_metrics

class RealisticBacktester:
    """
    Simulates out-of-sample portfolio backtesting incorporating transaction costs (bps/slippage),
    turnover tracking, rebalancing frequencies, and gross vs. net performance.
    """
    def __init__(self, prices_df, initial_capital=10000.0, transaction_cost_pct=0.0010, slippage_pct=0.0005, risk_free_rate=0.02):
        """
        prices_df: DataFrame of asset prices indexed by DatetimeIndex
        initial_capital: Starting investment capital ($)
        transaction_cost_pct: Fixed cost per trade (e.g. 0.0010 = 10 bps / 0.1%)
        slippage_pct: Execution slippage per trade (e.g. 0.0005 = 5 bps)
        risk_free_rate: Annualized risk-free interest rate
        """
        self.prices_df = prices_df.dropna()
        self.returns_df = np.log(self.prices_df / self.prices_df.shift(1)).dropna()
        self.initial_capital = initial_capital
        self.tc_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.total_cost_pct = transaction_cost_pct + slippage_pct
        self.rf_rate = risk_free_rate

    def run_backtest(self, weights, rebalance_freq='monthly', drift_threshold=0.05):
        """
        Runs portfolio simulation for target weights.
        rebalance_freq: 'monthly', 'quarterly', 'drift_trigger', or 'buy_and_hold'
        """
        asset_names = self.prices_df.columns
        w_series = pd.Series(weights)[asset_names]
        
        gross_returns = (self.returns_df * w_series).sum(axis=1)
        
        if rebalance_freq == 'monthly':
            try:
                rebalance_dates = self.returns_df.resample('ME').first().index
            except ValueError:
                rebalance_dates = self.returns_df.resample('M').first().index
        elif rebalance_freq == 'quarterly':
            try:
                rebalance_dates = self.returns_df.resample('QE').first().index
            except ValueError:
                rebalance_dates = self.returns_df.resample('Q').first().index
        else:
            rebalance_dates = [self.returns_df.index[0]]
            
        net_returns = gross_returns.copy()
        current_w = w_series.values
        total_turnover = 0.0
        
        for dt in self.returns_df.index:
            is_drift_trigger = (rebalance_freq == 'drift_trigger') and np.any(np.abs(current_w - w_series.values) > drift_threshold)
            
            if dt in rebalance_dates or is_drift_trigger:
                trade_turnover = np.sum(np.abs(w_series.values - current_w))
                total_turnover += trade_turnover
                tc_cost = trade_turnover * self.total_cost_pct
                net_returns.loc[dt] -= tc_cost
                current_w = w_series.values
            else:
                asset_rets = self.returns_df.loc[dt].values
                current_w = current_w * np.exp(asset_rets)
                if np.sum(current_w) > 0:
                    current_w /= np.sum(current_w)

                    current_w /= np.sum(current_w)
                    
        cum_gross = (1 + gross_returns).cumprod()
        cum_net = (1 + net_returns).cumprod()
        
        gross_metrics = calculate_comprehensive_risk_metrics(gross_returns, risk_free_rate=self.rf_rate)
        net_metrics = calculate_comprehensive_risk_metrics(net_returns, risk_free_rate=self.rf_rate)
        
        annualized_turnover = (total_turnover / (len(self.returns_df) / 252)) if len(self.returns_df) > 0 else 0.0
        
        return {
            "gross_cumulative": cum_gross,
            "net_cumulative": cum_net,
            "gross_returns": gross_returns,
            "net_returns": net_returns,
            "gross_metrics": gross_metrics,
            "net_metrics": net_metrics,
            "total_turnover": total_turnover,
            "annualized_turnover": annualized_turnover,
            "rebalance_freq": rebalance_freq
        }

def compare_strategy_vs_benchmark(prices_df, strategy_weights, benchmark_weights, tc_pct=0.0010, rf_rate=0.02):
    """
    Executes backtest comparing strategy portfolio against benchmark portfolio under net transaction costs.
    """
    bt = RealisticBacktester(prices_df, transaction_cost_pct=tc_pct, risk_free_rate=rf_rate)
    
    strat_res = bt.run_backtest(strategy_weights, rebalance_freq='monthly')
    bench_res = bt.run_backtest(benchmark_weights, rebalance_freq='monthly')
    
    summary_table = pd.DataFrame({
        "Strategy Portfolio (Net)": {
            "Total Return": f"{(strat_res['net_cumulative'].iloc[-1] - 1):.2%}",
            "Annualized Return": f"{strat_res['net_metrics']['Annualized Return']:.2%}",
            "Annualized Volatility": f"{strat_res['net_metrics']['Annualized Volatility']:.2%}",
            "Sharpe Ratio": f"{strat_res['net_metrics']['Sharpe Ratio']:.3f}",
            "Sortino Ratio": f"{strat_res['net_metrics']['Sortino Ratio']:.3f}",
            "Max Drawdown": f"{strat_res['net_metrics']['Max Drawdown']:.2%}",
            "95% VaR": f"{strat_res['net_metrics']['95% Value at Risk (VaR)']:.2%}",
            "95% CVaR": f"{strat_res['net_metrics']['95% Expected Shortfall (CVaR)']:.2%}",
            "Annualized Turnover": f"{strat_res['annualized_turnover']:.2%}"
        },
        "Benchmark Portfolio (60/40)": {
            "Total Return": f"{(bench_res['net_cumulative'].iloc[-1] - 1):.2%}",
            "Annualized Return": f"{bench_res['net_metrics']['Annualized Return']:.2%}",
            "Annualized Volatility": f"{bench_res['net_metrics']['Annualized Volatility']:.2%}",
            "Sharpe Ratio": f"{bench_res['net_metrics']['Sharpe Ratio']:.3f}",
            "Sortino Ratio": f"{bench_res['net_metrics']['Sortino Ratio']:.3f}",
            "Max Drawdown": f"{bench_res['net_metrics']['Max Drawdown']:.2%}",
            "95% VaR": f"{bench_res['net_metrics']['95% Value at Risk (VaR)']:.2%}",
            "95% CVaR": f"{bench_res['net_metrics']['95% Expected Shortfall (CVaR)']:.2%}",
            "Annualized Turnover": f"{bench_res['annualized_turnover']:.2%}"
        }
    })
    
    return summary_table, strat_res, bench_res

def run_stress_tests(strategy_weights, benchmark_weights):
    """
    Evaluates stress testing scenarios:
    1. Equity Market Selloff: SPY -20%, TSLA -35%, BND +2%
    2. Interest Rate Shock / Bond Selloff: BND -10%, SPY -5%, TSLA -8%
    3. Tech Crash & Volatility Spike: TSLA -40%, SPY -15%, BND +1%
    """
    scenarios = {
        "Severe Equity Selloff": {'TSLA': -0.35, 'SPY': -0.20, 'BND': 0.02},
        "Rate Shock / Bond Selloff": {'TSLA': -0.08, 'SPY': -0.05, 'BND': -0.10},
        "Tech Volatility Crash": {'TSLA': -0.40, 'SPY': -0.15, 'BND': 0.01}
    }
    
    stress_results = []
    for sc_name, shocks in scenarios.items():
        strat_impact = sum(strategy_weights[a] * shocks[a] for a in shocks)
        bench_impact = sum(benchmark_weights[a] * shocks[a] for a in shocks)
        stress_results.append({
            "Stress Scenario": sc_name,
            "Strategy Impact": f"{strat_impact:.2%}",
            "Benchmark Impact": f"{bench_impact:.2%}",
            "Downside Buffer": f"{(strat_impact - bench_impact):+.2%}"
        })
        
    return pd.DataFrame(stress_results)
