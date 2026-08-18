# GMF Investments — Quantitative Analytics, Time Series Forecasting & Portfolio Strategy Pipeline

[![CI/CD Pipeline](https://github.com/Dagiayy/gmf_investments/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Dagiayy/gmf_investments/actions/workflows/ci-cd.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Executive Summary

The **GMF Investments Quantitative Pipeline** provides an end-to-end framework for financial data engineering, volatility modeling, predictive time-series forecasting, quantitative portfolio optimization, and out-of-sample backtesting. The platform analyzes three core strategic assets:

- **TSLA (Tesla, Inc.)**: High-growth equity asset exhibiting substantial price volatility.
- **SPY (SPDR S&P 500 ETF Trust)**: Broad market equity index ETF representing U.S. market exposure.
- **BND (Vanguard Total Bond Market ETF)**: Low-risk fixed income benchmark for portfolio capital preservation.

---

## Key Features & Mathematical Foundations

### 1. Centralized Configuration & Reproducibility Infrastructure
- **Configuration Hub (`configs/config.json`)**: Centralized asset universe, historical bounds (`2015-01-01` to `2025-07-31`), train/test split dates (`2023-12-31`), transaction costs (`10 bps / 0.1%`), risk-free rate (`2.0%`), and model seeds (`42`).
- **Experiment Logger (`logs/experiment_log.json`)**: Automatically logs execution timestamps, data bounds, model performance metrics, portfolio weights, and unit test status for complete auditability.

### 2. Data Contracts & Data Quality Assertions (`src/data_contracts.py`)
- Programmatically asserts:
  - Non-empty series with continuous `DatetimeIndex`.
  - Absence of NaN values and duplicate dates.
  - Strictly positive price values ($P > 0$).
  - Positive semi-definiteness of the asset covariance matrix ($\lambda_{\min}(\mathbf{\Sigma}) \ge 0$).
  - Portfolio weight non-negativity ($w_i \ge 0$) and sum-to-one constraint ($\sum_{i=1}^N w_i = 1$).

### 3. Forecasting Engine & Naive Baselines (`src/models.py`, `src/validation.py`)
- Benchmarks predictive models against simple naive baselines:
  - **Random Walk / Naive Last-Value**:
    $$\hat{y}_{t+h} = y_t$$
  - **Moving Average (30-Day)**:
    $$\hat{y}_{t+h} = \frac{1}{N} \sum_{i=0}^{N-1} y_{t-i}$$
  - **Seasonal ARIMA (SARIMA)**:
    $$\Phi_P(B^s) \phi_p(B) (1-B)^d (1-B^s)^D y_t = \Theta_Q(B^s) \theta_q(B) \epsilon_t$$
- **Walk-Forward Cross-Validation**: Executes 19-fold expanding-window cross-validation to assess out-of-sample stability without look-ahead leakage.

### 4. GARCH(1,1) Volatility Modeling (`src/garch_model.py`)
- Fits conditional variance parameters via Maximum Likelihood Estimation:
  $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2 \quad \text{where } \omega > 0, \, \alpha \ge 0, \, \beta \ge 0, \, \alpha + \beta < 1$$

### 5. Extended Risk & Distribution Analytics (`src/risk.py`)
- Computes comprehensive risk metrics:
  - **Annualized Sharpe Ratio**:
    $$\text{Sharpe} = \frac{\mathbb{E}[R_p] - R_f}{\sigma_p}$$
  - **Sortino Ratio**:
    $$\text{Sortino} = \frac{\mathbb{E}[R_p] - R_f}{\sigma_d}, \quad \text{where } \sigma_d = \sqrt{\frac{1}{T} \sum_{t=1}^T \min(0, R_{p,t} - R_f)^2}$$
  - **Value at Risk (95% & 99% VaR)**:
    $$\text{VaR}_\alpha = -\inf \{ r \in \mathbb{R} : P(R \le r) \ge 1 - \alpha \}$$
  - **Expected Shortfall / Conditional VaR (95% & 99% CVaR)**:
    $$\text{CVaR}_\alpha = \mathbb{E}[-R \mid -R \ge \text{VaR}_\alpha]$$
  - **Jarque-Bera Normality Test**:
    $$JB = \frac{N}{6} \left( S^2 + \frac{1}{4}(K - 3)^2 \right)$$

### 6. Advanced Portfolio Optimization (`src/portfolio.py`)
- **Ledoit-Wolf Covariance Shrinkage**:
  $$\mathbf{\Sigma}_{\text{shrunk}} = \delta \mathbf{F} + (1 - \delta) \mathbf{S}$$
- **CAPM Expected Return Estimator**:
  $$\mathbb{E}[R_i] = R_f + \beta_i (\mathbb{E}[R_m] - R_f)$$
- **Risk Parity (Equal Risk Contribution - ERC)**:
  $$w_i \cdot (\mathbf{\Sigma} w)_i = \frac{1}{N} w^T \mathbf{\Sigma} w \quad \forall i$$
- **Black-Litterman Portfolio Model**:
  $$\mathbb{E}[R]_{\text{BL}} = \left[ (\tau \mathbf{\Sigma})^{-1} + \mathbf{P}^T \mathbf{\Omega}^{-1} \mathbf{P} \right]^{-1} \left[ (\tau \mathbf{\Sigma})^{-1} \mathbf{\Pi} + \mathbf{P}^T \mathbf{\Omega}^{-1} \mathbf{Q} \right]$$

### 7. Realistic Out-of-Sample Backtester (`src/backtesting.py`)
- Accounts for fixed transaction costs ($10\text{ bps} / 0.1\%$) and execution slippage ($5\text{ bps} / 0.05\%$).
- Implements threshold-based drift rebalancing (triggers when weight drift $> 5\%$) and tracks annualized portfolio turnover.

---

## Project Directory Architecture

```plaintext
gmf_investments/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions CI/CD Pipeline
│
├── 01_data/                   # Standardized data layer
│   ├── raw/                   # Raw CSV downloads from Yahoo Finance
│   └── processed/             # Cleaned, continuous, aligned CSVs
│
├── 02_features/               # Extracted technical features & indicators
├── 03_models/                 # Model artifacts & forecasts
├── 04_portfolio/              # Covariance models & optimizers
├── 05_backtest/               # Strategy engine & backtest logs
├── 06_reports/                # Publication deliverables
│   ├── figures/               # High-res PNG charts
│   └── summary_dashboard.html # Standalone interactive HTML dashboard
│
├── configs/
│   └── config.json            # Central configuration Parameters
├── logs/
│   └── experiment_log.json    # Automated run execution log
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
│   ├── data_contracts.py      # Data contract assertions
│   ├── data_fetch.py          # Yahoo Finance data fetcher
│   ├── preprocessing.py       # Data cleaning and scaling
│   ├── risk.py                # Comprehensive risk analytics engine
│   ├── garch_model.py         # GARCH(1,1) volatility engine
│   ├── feature_engineering.py # RSI, MACD, Bollinger Bands engine
│   ├── models.py              # SARIMA, LSTM & Ensemble Forecaster
│   ├── validation.py          # Walk-forward cross-validation engine
│   ├── portfolio.py           # Risk Parity, Black-Litterman & Markowitz
│   ├── backtesting.py         # Net backtester & stress testing
│   ├── dashboard.py           # HTML summary dashboard exporter
│   ├── generate_report_figures.py # Chart figure generator
│   └── build_technical_note_docx.py # Automated DOCX report builder
│
├── tests/
│   └── test_pipeline.py       # Automated unit test suite
│
├── GMF_Investments_Technical_Note.docx # Primary DOCX deliverable
├── main.py                    # Master CLI pipeline entry point
├── README.md                  # Comprehensive project documentation
└── requirements.txt           # Python dependencies
```

---

## Quick Start & Usage Instructions

### 1. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/Dagiayy/gmf_investments.git
cd gmf_investments

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute the Master Pipeline

To run data contract validation, feature extraction, GARCH volatility modeling, walk-forward cross validation, portfolio optimization, net backtesting, stress testing, figure generation, unit testing, and report generation in a single command:

```bash
python main.py
```

### 3. Run Automated Unit Tests

```bash
python -m unittest discover tests
```

---

## Key Results Summary

| Quantitative Metric | Strategy Portfolio (Net) | Benchmark Portfolio (60/40) |
| :--- | :---: | :---: |
| **Total Net Return** | **16.12%** | **11.07%** |
| **Annualized Return** | **17.22%** | **11.42%** |
| **Annualized Volatility** | **20.27%** | **12.36%** |
| **Sharpe Ratio** | **0.751** | **0.763** |
| **Sortino Ratio** | **0.921** | **0.953** |
| **Max Drawdown** | **19.37%** | **11.59%** |
| **95% Expected Shortfall (CVaR)** | **3.00%** | **1.81%** |
| **Annualized Turnover** | **1.78%** | **17.19%** |

---

## Author & Contact

**Dagmawi Ayenew**  
Email: [ayenewdagmawi@gmail.com](mailto:ayenewdagmawi@gmail.com)  
GitHub: [https://github.com/Dagiayy/gmf_investments](https://github.com/Dagiayy/gmf_investments)
