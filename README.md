
# GMF Investments — Financial Data Analytics & Forecasting Pipeline

## Project Overview

This project focuses on financial time series data analysis and forecasting for three major assets:

* **TSLA (Tesla, Inc.)** — high growth and volatility
* **BND (Vanguard Total Bond Market ETF)** — low risk, stable bonds
* **SPY (SPDR S\&P 500 ETF Trust)** — diversified equity exposure

The goal is to preprocess and explore financial data (**Task 1**) and build predictive models (**Task 2**) to forecast future stock prices. This serves as a foundation for quantitative investment strategies and risk management.

---

## Tasks

### Task 1: Data Preprocessing and Exploratory Data Analysis (EDA)

* **Download** raw historical market data using Yahoo Finance API.
* **Clean** data to handle missing values, fix inconsistencies, and ensure proper formats.
* **Visualize** price trends, daily returns, rolling statistics, and volatility.
* **Perform** stationarity tests (ADF test) to verify suitability for time series modeling.
* **Calculate** key risk metrics: Value at Risk (VaR) and Sharpe Ratio.
* **Output** cleaned datasets and summary visualizations for further analysis.

---

### Task 2: Time Series Forecasting Models

* **Split data** chronologically into training (2015-2023) and testing (2024-2025) to preserve time order.
* **Develop and optimize** classical models: ARIMA and SARIMA, with grid search for parameters.
* **Build and train** a deep learning model: LSTM neural network.
* **Forecast** stock prices on the test set.
* **Evaluate** models using performance metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and optionally MAPE.
* **Compare** model results and discuss trade-offs between interpretability and accuracy.

---

### Task 3: Forecast Future Market Trends

* **Generate** future price predictions for 6-12 months using the best trained model(s).
* **Visualize** forecasts alongside historical prices with confidence intervals.
* **Interpret** long-term trends, identifying upward/downward movements or anomalies.
* **Analyze** forecast uncertainty by examining confidence interval widths over the horizon.
* **Outline** market opportunities and risks based on forecast trends and volatility.

---

### Task 4: Portfolio Optimization Based on Forecast

* **Use** forecasted return for TSLA from the best-performing model as expected return.
* **Use** historical average returns for stable assets (BND, SPY).
* **Compute** covariance matrix from historical returns of all assets.
* **Run** portfolio optimization based on Modern Portfolio Theory.
* **Generate** and plot the Efficient Frontier.
* **Identify** and mark Maximum Sharpe Ratio and Minimum Volatility portfolios.
* **Recommend** an optimal portfolio with expected return, volatility, and weights.

---

### Task 5: Strategy Backtesting (Final Task)

* **Define** backtesting period using the last year of data (e.g., Aug 1, 2024 – Jul 31, 2025).
* **Establish** a benchmark portfolio (e.g., static 60% SPY / 40% BND).
* **Simulate** the strategy portfolio performance based on optimal weights from Task 4.
* **Hold** portfolio weights fixed or perform simplified rebalancing over the period.
* **Calculate** cumulative returns, total return, and annualized Sharpe Ratio.
* **Compare** the strategy performance against the benchmark.
* **Analyze** and conclude on strategy viability and risk-adjusted performance.

---

## Technologies and Libraries

* Python 3.8+
* pandas, numpy — Data manipulation and numerical operations
* matplotlib, seaborn — Visualization
* statsmodels — Statistical tests, ARIMA/SARIMA modeling
* scikit-learn — Data preprocessing and evaluation metrics
* TensorFlow / Keras — LSTM modeling and training

---

## Project Structure

```plaintext
gmf_investments/
│
├── data/
│   ├── raw/                  # Raw CSVs downloaded from Yahoo Finance
│   ├── processed/            # Cleaned and preprocessed CSVs
│
├── notebooks/
│   ├── 01_data_fetch_and_clean.ipynb    # Task 1: data fetching & cleaning
│   ├── 02_eda.ipynb                      # Task 1: exploratory data analysis
│   ├── 03_model_arima_lstm.ipynb         # Task 2: time series modeling & comparison
│   ├── 04_forecasting.ipynb              # Task 2 & 3: forecasting & trend analysis
│   ├── 05_portfolio_optimization.ipynb   # Task 4: portfolio optimization
│   ├── 06_backtesting.ipynb               # Task 5: strategy backtesting
│
├── src/
│   ├── data_fetch.py          # Functions to download data from Yahoo Finance
│   ├── preprocessing.py       # Data cleaning and feature engineering functions
│   ├── eda.py                 # Plotting and summary statistics
│   ├── models.py              # ARIMA/SARIMA & LSTM model implementations
│   ├── portfolio.py           # Portfolio optimization functions
│   ├── backtesting.py         # Backtesting logic
│   └── utils.py               # Helper functions
│
├── reports/
│   ├── interim_report.pdf     # Interim analysis report
│   ├── final_investment_memo.pdf  # Final investment memo
│   └── figures/               # Visualizations (price trends, model diagnostics)
│
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── main.py                   # Optional: run full pipeline script
````

---

## Usage Instructions

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Run Data Preprocessing (Task 1)

```bash
python src/task1_preprocess.py
```

### 3. Run Exploratory Data Analysis (Task 1)

```bash
python src/task1_eda.py
```

### 4. Build and Evaluate Forecasting Models (Task 2)

```bash
python src/task2_modeling.py
```

### 5. Forecast Future Market Trends (Task 3)

Use the Jupyter notebook:

* `notebooks/04_forecasting.ipynb`

### 6. Portfolio Optimization (Task 4)

Use the Jupyter notebook:

* `notebooks/05_portfolio_optimization.ipynb`

### 7. Strategy Backtesting (Task 5)

Use the Jupyter notebook:

* `notebooks/06_backtesting.ipynb`

---

## Results Summary

* **Data Cleaning:** Missing values handled, dates properly formatted, and consistent time series constructed.

* **EDA:** TSLA showed high volatility and non-stationarity in prices; returns were more stationary.

* **ARIMA/SARIMA:** Identified best seasonal parameters with grid search (e.g., SARIMA(0,1,1)x(0,1,1,12)) for Tesla price forecasting.

* **LSTM:** Deep learning model showed lower MAE and RMSE compared to ARIMA, indicating improved accuracy but higher complexity.

* **Forecasting & Trend Analysis:**

  * Generated 6-12 month forecasts with confidence intervals.
  * Observed general upward trend with widening confidence intervals over time.
  * Highlighted increasing uncertainty in long-term forecasts.
  * Provided actionable insights on market opportunities and risks.

* **Portfolio Optimization:**

  * Used forecasted returns and historical data to compute efficient frontier.
  * Identified Maximum Sharpe Ratio and Minimum Volatility portfolios.
  * Recommended an optimal portfolio with asset weights and risk-return profile.

* **Backtesting:**

  * Simulated strategy portfolio performance against a benchmark (60% SPY / 40% BND).
  * Strategy achieved a total return of \~9.1% with Sharpe ratio \~0.97.
  * Benchmark achieved a total return of \~11.1% with Sharpe ratio \~0.93.
  * Strategy showed better risk-adjusted return, suggesting viability despite lower raw return.
  * Backtest highlights the importance of risk management alongside return maximization.

---

## Future Directions

* Expand forecasting models with additional techniques (GARCH, Prophet).
* Integrate macroeconomic variables and sentiment data for enriched features.
* Implement more sophisticated portfolio optimization and risk budgeting.
* Add realistic backtesting features like transaction costs, rebalancing, and drawdowns.
* Automate full data pipeline and enable near real-time forecasting and trading signals.

---

## Contact

**Dagmawi Ayenew**
Email: [ayenewdagmawi@gmail.com](mailto:ayenewdagmawi@gmail.com)

```


