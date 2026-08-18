import os
import json
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# Pipeline module imports
from src.data_contracts import validate_price_df, validate_aligned_returns, validate_portfolio_weights
from src.preprocessing import load_data, fill_missing
from src.feature_engineering import extract_asset_features
from src.risk import calculate_comprehensive_risk_metrics
from src.garch_model import fit_asset_garch_volatility, GARCHModel
from src.models import (
    load_series, train_test_split_series, naive_last_value_forecast, 
    moving_average_forecast, compute_metrics, perform_residual_diagnostics,
    EnsembleForecaster
)
from src.validation import WalkForwardValidator
from src.portfolio import (
    optimize_max_sharpe, optimize_min_volatility, optimize_risk_parity, 
    optimize_black_litterman, run_sensitivity_analysis, 
    estimate_shrinkage_covariance, estimate_capm_expected_returns
)
from src.backtesting import RealisticBacktester, compare_strategy_vs_benchmark, run_stress_tests
from src.dashboard import export_html_summary_dashboard
from src.generate_report_figures import main as generate_figures
from src.build_technical_note_docx import build_docx

def load_config(config_path="configs/config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_experiment_log(log_data, log_path="logs/experiment_log.json"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

def run_pipeline():
    print("==========================================================================")
    print("  GMF INVESTMENTS: QUANTITATIVE PIPELINE & EXTENSION ROADMAP EXECUTION")
    print("==========================================================================")
    
    # 1. Load Configuration & Experiment Tracking Setup
    config = load_config()
    start_time = datetime.now().isoformat()
    print("[OK] Phase 1: Loaded central configuration from configs/config.json")
    
    # 2. Data Engineering & Data Contracts
    assets = [a.lower() for a in config["assets"]]
    raw_dir = config["data_paths"]["raw_dir"]
    proc_dir = config["data_paths"]["processed_dir"]
    
    prices_dict = {}
    for asset in assets:
        path = os.path.join(proc_dir, f"{asset}_processed.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        validate_price_df(df, asset_name=asset.upper())
        prices_dict[asset.upper()] = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        
        # Feature Engineering Extraction
        extract_asset_features(df, asset_name=asset.upper())
        
    prices_df = pd.DataFrame(prices_dict).dropna()
    returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
    validate_aligned_returns(returns_df)
    print("[OK] Phase 2: Data contracts validated & technical features extracted.")
    
    # 3. GARCH(1,1) Volatility Modeling
    print("\n--- GARCH(1,1) Dynamic Volatility Modeling ---")
    garch_vols = fit_asset_garch_volatility(returns_df)
    print("Recent GARCH Annualized Volatility Estimates:\n", garch_vols.tail())
    
    # 4. Risk Analytics & EDA Profiling
    print("\n--- Asset Risk Analytics & Distribution Profiling ---")
    risk_summary = {}
    for asset in config["assets"]:
        market_ret = returns_df["SPY"] if asset != "SPY" else None
        risk_summary[asset] = calculate_comprehensive_risk_metrics(
            returns_df[asset], market_returns=market_ret, risk_free_rate=config["portfolio"]["risk_free_rate"]
        )
    df_risk = pd.DataFrame(risk_summary)
    print(df_risk.to_string())
    
    # 5. Naive Baselines & Ensemble Forecasting
    print("\n--- Naive Baseline & Ensemble Forecasting ---")
    wfv = WalkForwardValidator(initial_train_size=1260, horizon=126, step_size=63)
    wfv_res = wfv.evaluate_baselines(prices_df["TSLA"])
    print(f"Evaluated {wfv_res['Num_Folds']} expanding-window folds for TSLA:")
    print("Naive Last-Value Metrics:", wfv_res["Naive_Last_Value"])
    print("Moving Average (30D) Metrics:", wfv_res["Moving_Average_30D"])
    
    ensemble = EnsembleForecaster()
    tsla_train = prices_df["TSLA"].loc[:config["dates"]["train_end"]]
    ens_res = ensemble.predict(tsla_train, steps=126)
    print("Ensemble Forecast Start Price:", ens_res["ensemble_forecast"][0])
    print("Ensemble Forecast End Price:", ens_res["ensemble_forecast"][-1])
    
    # 6. Advanced Portfolio Optimization Suite
    print("\n--- Portfolio Optimization & Covariance Shrinkage ---")
    exp_returns = pd.Series({'TSLA': 0.17337, 'BND': 0.018088, 'SPY': 0.128134})
    sample_cov = returns_df.cov() * 252
    shrunk_cov = estimate_shrinkage_covariance(returns_df)
    capm_returns = estimate_capm_expected_returns(returns_df)
    
    print("CAPM Expected Return Estimates:\n", capm_returns)
    
    opt_max_s = optimize_max_sharpe(exp_returns, sample_cov, risk_free_rate=config["portfolio"]["risk_free_rate"])
    opt_min_v = optimize_min_volatility(sample_cov)
    opt_rp = optimize_risk_parity(sample_cov)
    
    market_caps = {'TSLA': 800e9, 'SPY': 500e9, 'BND': 300e9}
    views = {'TSLA': 0.15}
    opt_bl = optimize_black_litterman(market_caps, sample_cov, views, tau=config["portfolio"]["black_litterman_tau"])
    
    print("Max Sharpe Weights:", opt_max_s["weights"])
    print("Risk Parity Weights:", opt_rp["weights"])
    print("Black-Litterman Weights:", opt_bl["weights"])
    
    print("\n--- Expected Return Sensitivity Analysis ---")
    sens_df = run_sensitivity_analysis(exp_returns, sample_cov, target_asset='TSLA')
    print(sens_df.to_string())
    
    # 7. Realistic Out-of-Sample Backtesting & Stress Testing
    print("\n--- Out-of-Sample Realistic Backtesting (Net of Costs & Slippage) ---")
    bt_start = config["dates"]["backtest_start"]
    bt_end = config["dates"]["backtest_end"]
    bt_prices = prices_df.loc[bt_start:bt_end]
    
    strat_weights = opt_max_s["weights"]
    bench_weights = config["backtest"]["benchmark_weights"]
    
    summary_bt, strat_res, bench_res = compare_strategy_vs_benchmark(
        bt_prices, strat_weights, bench_weights, 
        tc_pct=config["backtest"]["transaction_cost_pct"],
        rf_rate=config["backtest"]["risk_free_rate"]
    )
    print(summary_bt.to_string())
    
    print("\n--- Macro/Market Stress Testing Scenarios ---")
    stress_df = run_stress_tests(strat_weights, bench_weights)
    print(stress_df.to_string())
    
    # 8. Generate Reports & Dashboard Exporter
    print("\n--- Generating Report Figures & HTML Dashboard ---")
    generate_figures()
    export_html_summary_dashboard(df_risk, summary_bt, sens_df)
    
    # 9. Run Unit Tests
    print("\n--- Running Automated Unit Test Suite ---")
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=1)
    test_res = runner.run(suite)
    if not test_res.wasSuccessful():
        raise RuntimeError("Unit test suite failed! Check logs.")
    print("[OK] Automated unit tests passed successfully.")
    
    # 10. Build Technical Note DOCX Document
    print("\n--- Building Technical Note DOCX Document ---")
    build_docx()
    
    # 11. Record Experiment Metadata Log
    log_payload = {
        "timestamp": start_time,
        "config": config,
        "data_summary": {
            "num_rows": len(prices_df),
            "start_date": str(prices_df.index[0]),
            "end_date": str(prices_df.index[-1])
        },
        "wfv_baselines": wfv_res,
        "portfolio_allocations": {
            "max_sharpe": opt_max_s["weights"],
            "risk_parity": opt_rp["weights"],
            "black_litterman": opt_bl["weights"]
        },
        "backtest_summary": summary_bt.to_dict(),
        "unit_tests_passed": test_res.wasSuccessful()
    }
    save_experiment_log(log_payload)
    print("[OK] Saved experiment run log to logs/experiment_log.json")
    
    print("\n==========================================================================")
    print("  [OK] PIPELINE & ROADMAP EXECUTION COMPLETE SUCCESSFULLY!")
    print("==========================================================================")

if __name__ == '__main__':
    run_pipeline()
