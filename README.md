

# GMF Investments — Financial Data Preprocessing and Exploratory Analysis

## Project Overview

This project aims to preprocess and conduct exploratory data analysis (EDA) on historical financial data for three major assets: TSLA (Tesla, Inc.), BND (Vanguard Total Bond Market ETF), and SPY (SPDR S\&P 500 ETF Trust). These assets represent a balanced mix of high-growth/high-volatility, stable/low-risk, and diversified/moderate-risk investment vehicles.

The primary objective is to clean, understand, and analyze the financial time series data as a foundational step toward advanced modeling, risk assessment, and investment strategy development.

---

## Motivation

In financial analytics and quantitative investing, thorough data preprocessing and robust exploratory analysis are critical to ensure data quality and to derive actionable insights. This enables better forecasting, portfolio optimization, and risk management.

---

## Project Scope

* **Data Acquisition:** Download historical market data using Yahoo Finance API.
* **Data Cleaning:** Handle missing values, ensure consistent data types, and prepare datasets for analysis.
* **Statistical Summary:** Generate descriptive statistics to understand asset behavior and data distribution.
* **Normalization:** Scale features as needed for subsequent machine learning or statistical modeling.
* **Exploratory Data Analysis:**

  * Visualize price trends over time to identify patterns and regimes.
  * Compute and plot daily returns and their distributions to gauge volatility.
  * Analyze rolling statistics (mean and standard deviation) to assess short-term trends and fluctuations.
  * Detect outliers and significant anomalies that may impact modeling and decision-making.
* **Stationarity Testing:** Apply Augmented Dickey-Fuller (ADF) tests to detect stationarity in prices and returns — a prerequisite for many time series models.
* **Risk Assessment:** Calculate foundational risk metrics including Value at Risk (VaR) and Sharpe Ratio for evaluating potential losses and risk-adjusted returns.

---

## Data Description

* **TSLA (Tesla, Inc.):** High volatility, growth-oriented stock with potential for significant returns and risks.
* **BND (Vanguard Total Bond Market ETF):** Stable bond market exposure, providing low volatility and risk mitigation.
* **SPY (SPDR S\&P 500 ETF Trust):** Broad market exposure representing diversified equities with moderate risk and return profiles.

---

## Technologies and Libraries

* **Python 3.8+** — Core programming language
* **pandas** — Data manipulation and cleaning
* **numpy** — Numerical computations
* **matplotlib & seaborn** — Data visualization
* **statsmodels** — Statistical tests (e.g., ADF test)
* **scikit-learn** — Data scaling and preprocessing

---

## Usage Guide

1. **Environment Setup**
   Create and activate a virtual environment, then install required libraries:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

2. **Data Preprocessing**
   Run the preprocessing script to clean raw data, fill missing values, and save processed datasets:

   ```bash
   python src/task1_preprocess.py
   ```

3. **Exploratory Data Analysis**
   Execute the EDA script to generate plots, perform statistical tests, and calculate risk metrics:

   ```bash
   python src/task1_eda.py
   ```

4. **Outputs**

   * Processed datasets saved under `data/processed/`
   * Visualizations and reports saved under `output/` folder for review

---

## Key Deliverables

* Cleaned and processed financial time series data ready for modeling
* Comprehensive exploratory visualizations including:

  * Price trends
  * Daily percentage returns
  * Rolling volatility measures
  * Outlier detection reports
* Statistical test results for stationarity (ADF tests) with interpretations
* Risk metrics summarizing historical return distributions and risk-adjusted performance

---

## Insights and Implications

* TSLA exhibits significant volatility and non-stationary price behavior, requiring differencing or transformation for time series modeling.
* Daily returns tend to be stationary, supporting modeling approaches on returns rather than raw prices.
* Detected multiple outliers correlating with notable market events, suggesting the importance of anomaly detection in risk management.
* Risk metrics such as VaR and Sharpe Ratio provide quantitative benchmarks for potential losses and investment attractiveness.

---

## Future Work

* Implement and evaluate predictive time series models (ARIMA, GARCH, LSTM).
* Expand analysis to multi-asset portfolio optimization and risk budgeting.
* Incorporate macroeconomic indicators and sentiment analysis to enrich modeling.
* Automate data pipeline and integrate with real-time data feeds.

---

## Project Structure

```
gmf_investments/
│
├── data/                        # Raw & processed datasets
│   ├── raw/
│   │   ├── tsla_raw.csv
│   │   ├── bnd_raw.csv
│   │   └── spy_raw.csv
│   ├── processed/
│   │   ├── tsla_processed.csv
│   │   ├── bnd_processed.csv
│   │   └── spy_processed.csv
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── 01_data_fetch_and_clean.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_arima_lstm.ipynb
│   ├── 04_forecasting.ipynb
│   ├── 05_portfolio_optimization.ipynb
│   └── 06_backtesting.ipynb
│
├── src/                         # Python scripts for reusable code
│   ├── __init__.py
│   ├── data_fetch.py             # yfinance fetching functions
│   ├── preprocessing.py          # cleaning, feature engineering
│   ├── eda.py                    # plotting and descriptive stats
│   ├── models.py                 # ARIMA, LSTM implementations
│   ├── portfolio.py              # MPT & efficient frontier
│   ├── backtesting.py            # backtest logic
│   └── utils.py                  # helpers
│
├── reports/                      # Outputs for submission
│   ├── interim_report.pdf
│   ├── final_investment_memo.pdf
│   └── figures/
│       ├── tsla_trend.png
│       ├── efficient_frontier.png
│       └── backtest_results.png
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview & instructions
└── main.py                       # Optional script to run full pipeline

```

---

## Contact

**Dagmawi Ayenew**
Email: \[[ayenewdagmawi@gmail.com]]


---
