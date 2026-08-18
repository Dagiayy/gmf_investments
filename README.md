# GMF Investments — Quantitative Analytics, Time Series Forecasting & Portfolio Strategy Pipeline

## Project Overview

This repository houses the **GMF Investments Quantitative Pipeline & Roadmap Implementation**. The codebase performs financial data engineering, exploratory data analysis, time-series forecasting (ARIMA/SARIMA vs Deep Learning LSTM), Modern Portfolio Theory optimization (Markowitz, Risk Parity, Black-Litterman), and net-of-transaction-cost backtesting across three strategic asset classes:

* **TSLA (Tesla, Inc.)** — High-growth equity with significant volatility.
* **SPY (SPDR S&P 500 ETF Trust)** — Broad market U.S. index benchmark.
* **BND (Vanguard Total Bond Market ETF)** — Fixed income asset for risk stabilization.

---

## Technical Highlights & Roadmap Enhancements

### 1. Central Configuration & Reproducibility Infrastructure
* **Central Config (`configs/config.json`)**: All asset universe settings, date bounds (2015–2025), train/test split dates (`2023-12-31`), backtest period (`2024-08-01` to `2025-07-31`), transaction cost assumptions (`10 bps / 0.1%`), risk-free rate (`2.0%`), and model seeds (`42`) are centralized.
* **Experiment Logging (`logs/experiment_log.json`)**: Automatically logs execution timestamps, dataset bounds, model metrics, portfolio weights, and test results for full auditability.

### 2. Data Contracts & Leakage Prevention (`src/data_contracts.py`)
* Enforces data contract assertions: continuous `DatetimeIndex`, non-empty series, zero NaN occurrences, non-positive price checks ($P > 0$), positive semi-definite covariance matrices ($\lambda_{\min} \ge 0$), and portfolio weight sum-to-1 constraints ($\sum w_i = 1$).

### 3. Naive Forecasting Baselines & Walk-Forward Validation Engine (`src/validation.py`)
* Benchmarks complex models against simple naive baselines: **Random Walk / Naive Last-Value** ($y_{t+h} = y_t$) and **30-Day Moving Average**.
* Implements a 19-fold expanding-window `WalkForwardValidator` to evaluate model stability across multiple historical market regimes without data leakage.

### 4. Comprehensive Risk Analytics (`src/risk.py`)
* Extends basic VaR and Sharpe ratios with:
  * **Sortino Ratio** (downside deviation risk-adjusted return)
  * **Calmar Ratio** (annualized return / max drawdown)
  * **Maximum Drawdown (MDD)** & Peak-to-Trough duration
  * **95% & 99% Expected Shortfall / Conditional VaR (CVaR)**
  * **Jarque-Bera Normality Test** & higher-order moments (Skewness, Kurtosis)
  * **Asset Beta** relative to SPY market index.

### 5. Advanced Portfolio Optimization (`src/portfolio.py`)
* **Ledoit-Wolf Covariance Shrinkage**: Reduces sample noise in asset covariance estimation.
* **CAPM Expected Returns**: Estimates CAPM expected returns ($E(R_i) = R_f + \beta_i (E(R_m) - R_f)$).
* **Advanced Optimizers**: Supports **Markowitz Maximum Sharpe & Minimum Volatility**, **Risk Parity (Equal Risk Contribution - ERC)**, **Black-Litterman Allocation**, and **Return Sensitivity Stress Testing**.

### 6. Realistic Strategy Backtesting & Stress Scenarios (`src/backtesting.py`)
* Simulates out-of-sample portfolio performance net of $10\text{ bps}$ ($0.1\%$) transaction costs per rebalance trade.
* Tracks annualized portfolio turnover and compares monthly vs. quarterly rebalancing schedules.
* **Macro Stress Testing**: Evaluates portfolio downside buffers under *Severe Equity Selloffs*, *Rate Shocks*, and *Tech Volatility Crashes*.

### 7. Automated Unit Test Suite (`tests/test_pipeline.py`)
* Unit tests validating data contract assertions, risk calculations, covariance shrinkage, CAPM returns, portfolio constraints, and backtest transaction cost accounting.

---

## Repository Architecture

```plaintext
gmf_investments/
│
├── 01_data/                 # Standardized raw and processed datasets
│   ├── raw/                 # Raw market CSVs from Yahoo Finance
│   └── processed/           # Cleaned, continuous, aligned CSVs
│
├── 02_features/             # Derived return series, rolling indicators
├── 03_models/               # Model artifacts (ARIMA/SARIMA, LSTM weights)
├── 04_portfolio/            # Covariance models & portfolio optimizers
├── 05_backtest/             # Strategy backtester & transaction cost logic
├── 06_reports/              # Generated publication figures & reports
│   └── figures/             # High-resolution PNG plots
│
├── configs/
│   └── config.json          # Central configuration parameters
├── logs/
│   └── experiment_log.json  # Automated execution run log
│
├── notebooks/
│   ├── 01_data_fetch_and_clean.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_arima_lstm.ipynb
│   ├── 04_forecasting.ipynb
│   ├── 05_portfolio_optimization.ipynb
│   └── 06_backtesting.ipynb
│
├── src/
│   ├── data_contracts.py    # Schema & data contract assertions
│   ├── data_fetch.py        # Yahoo Finance data fetcher
│   ├── preprocessing.py     # Data cleaning and normalization
│   ├── risk.py              # Extended risk analytics engine
│   ├── models.py            # SARIMA, LSTM & Naive forecasting baselines
│   ├── validation.py        # Walk-forward cross-validation engine
│   ├── portfolio.py         # Risk Parity, Black-Litterman & Markowitz
│   ├── backtesting.py       # Realistic net backtester & stress testing
│   ├── generate_report_figures.py # Plot figure generator
│   └── build_technical_note_docx.py # Automated DOCX report builder
│
├── tests/
│   └── test_pipeline.py     # Automated unit test suite
│
├── GMF_Investments_Technical_Note.docx # Publication DOCX report
├── main.py                  # Single master CLI entry point
├── README.md                # Project documentation
└── requirements.txt         # Dependencies
```

---

## Quick Start & Usage Instructions

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline

To execute data contract checks, EDA risk analytics, walk-forward cross validation, portfolio optimization, realistic backtesting, unit tests, and automated DOCX report generation in a single command:

```bash
python main.py
```

### 3. Run Automated Unit Tests

```bash
python -m unittest discover tests
```

---

## Results Summary

* **Data Integrity**: Enforced strict data contracts across 10 years of market data; validated positive semi-definite covariance matrices.
* **Forecasting**: Deep Learning LSTM achieved $9.15\%$ MAPE on test data vs. $13.24\%$ for SARIMA and $24.62\%$ for Naive Last-Value baseline.
* **Portfolio Optimization**:
  * **Max Sharpe (Markowitz)**: 55.34% BND / 44.66% SPY / 0.00% TSLA.
  * **Risk Parity (ERC)**: 75.24% BND / 18.74% SPY / 6.02% TSLA.
  * **Black-Litterman**: 86.68% TSLA / 13.32% SPY / 0.00% BND.
* **Backtesting (Net of 10 bps Costs)**: Strategy portfolio achieved an out-of-sample Sharpe Ratio of **0.751** net of costs vs. benchmark **0.763**, with significantly lower drawdown and downside risk exposure.

---

## Contact & Author

**Dagmawi Ayenew**  
Email: [ayenewdagmawi@gmail.com](mailto:ayenewdagmawi@gmail.com)  
GitHub: [https://github.com/Dagiayy/gmf_investments](https://github.com/Dagiayy/gmf_investments)
