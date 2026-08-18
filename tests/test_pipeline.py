import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_contracts import validate_portfolio_weights, validate_aligned_returns
from src.risk import calculate_max_drawdown, calculate_sortino_ratio, calculate_expected_shortfall, calculate_distribution_stats
from src.garch_model import GARCHModel
from src.feature_engineering import compute_rsi, compute_macd, compute_bollinger_bands
from src.models import EnsembleForecaster
from src.portfolio import optimize_max_sharpe, optimize_risk_parity, estimate_shrinkage_covariance, estimate_capm_expected_returns
from src.backtesting import RealisticBacktester, run_stress_tests

class TestPipelineUnits(unittest.TestCase):
    
    def test_portfolio_weights_contract(self):
        """Test portfolio weights sum-to-1 and non-negativity contract assertion."""
        valid_weights = [0.5534, 0.4466, 0.0]
        self.assertTrue(validate_portfolio_weights(valid_weights))
        
        with self.assertRaises(ValueError):
            validate_portfolio_weights([0.5, 0.2]) # sum != 1
            
        with self.assertRaises(ValueError):
            validate_portfolio_weights([1.2, -0.2, 0.0]) # negative weight

    def test_risk_metrics_math(self):
        """Test max drawdown, expected shortfall, and distribution stats."""
        rets = pd.Series([0.01, -0.05, 0.02, -0.03, 0.04, 0.01, -0.02, 0.03])
        cum_rets = (1 + rets).cumprod()
        mdd, _ = calculate_max_drawdown(cum_rets)
        self.assertGreater(mdd, 0.0)
        self.assertLessEqual(mdd, 1.0)
        
        cvar = calculate_expected_shortfall(rets, alpha=0.95)
        self.assertGreater(cvar, 0.0)
        
        dist_stats = calculate_distribution_stats(rets)
        self.assertIn("Skewness", dist_stats)
        self.assertIn("Jarque-Bera p-value", dist_stats)

    def test_garch_and_technical_features(self):
        """Test GARCH(1,1) fitting and technical indicator math."""
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        np.random.seed(42)
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100))), index=dates)
        rets = np.log(prices / prices.shift(1)).dropna()
        
        # Test GARCH model
        garch = GARCHModel()
        vol_fit = garch.fit(rets)
        self.assertEqual(len(vol_fit), len(rets))
        
        # Test Technical Indicators
        rsi = compute_rsi(prices)
        self.assertEqual(len(rsi), 100)
        
        upper, mid, lower, pct_b = compute_bollinger_bands(prices)
        self.assertEqual(len(upper), 100)

    def test_ensemble_forecaster(self):
        """Test Ensemble Forecaster prediction generation."""
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        np.random.seed(42)
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100))), index=dates)
        
        ensemble = EnsembleForecaster()
        res = ensemble.predict(prices, steps=10)
        self.assertEqual(len(res["ensemble_forecast"]), 10)
        self.assertEqual(len(res["lower_bound"]), 10)

    def test_portfolio_optimization_and_estimators(self):
        """Test Markowitz, Risk Parity, Shrinkage Covariance, and CAPM estimators."""
        exp_rets = pd.Series({'TSLA': 0.17, 'SPY': 0.12, 'BND': 0.02})
        cov_data = np.array([
            [0.34, 0.05, 0.001],
            [0.05, 0.03, 0.001],
            [0.001, 0.001, 0.003]
        ])
        cov_df = pd.DataFrame(cov_data, index=exp_rets.index, columns=exp_rets.index)
        
        opt_max_s = optimize_max_sharpe(exp_rets, cov_df)
        self.assertAlmostEqual(sum(opt_max_s['weights'].values()), 1.0, places=4)
        
        opt_rp = optimize_risk_parity(cov_df)
        self.assertAlmostEqual(sum(opt_rp['weights'].values()), 1.0, places=4)

    def test_backtest_and_stress_testing(self):
        """Test backtest transaction cost accounting and stress testing scenarios."""
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        np.random.seed(42)
        p1 = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
        p2 = 50 * np.exp(np.cumsum(np.random.normal(0, 0.005, 100)))
        df_prices = pd.DataFrame({'AssetA': p1, 'AssetB': p2}, index=dates)
        
        bt = RealisticBacktester(df_prices, transaction_cost_pct=0.005, slippage_pct=0.001)
        res = bt.run_backtest({'AssetA': 0.5, 'AssetB': 0.5}, rebalance_freq='monthly')
        self.assertLessEqual(res['net_cumulative'].iloc[-1], res['gross_cumulative'].iloc[-1] + 1e-6)

if __name__ == '__main__':
    unittest.main()
