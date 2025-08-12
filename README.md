
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

## Technologies and Libraries

* Python 3.8+
* pandas, numpy — Data manipulation and numerical operations
* matplotlib, seaborn — Visualization
* statsmodels — Statistical tests, ARIMA/SARIMA modeling
* scikit-learn — Data preprocessing and evaluation metrics
* TensorFlow / Keras — LSTM modeling and training

---

## Project Structure

```
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
│   ├── 04_forecasting.ipynb              # Task 2: forecasting & evaluation
│
├── src/
│   ├── data_fetch.py          # Functions to download data from Yahoo Finance
│   ├── preprocessing.py       # Data cleaning and feature engineering functions
│   ├── eda.py                 # Plotting and summary statistics
│   ├── models.py              # ARIMA/SARIMA & LSTM model implementations
│   ├── portfolio.py           # Portfolio optimization (future work)
│   ├── backtesting.py         # Backtesting logic (future work)
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
```

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

This step fetches raw data, cleans it, and saves processed datasets.

```bash
python src/task1_preprocess.py
```

### 3. Run Exploratory Data Analysis (Task 1)

Generates visualizations and statistics for understanding the data.

```bash
python src/task1_eda.py
```

### 4. Build and Evaluate Forecasting Models (Task 2)

Fits ARIMA/SARIMA and LSTM models, generates forecasts, and evaluates performance.

```bash
python src/task2_modeling.py
```

Or explore the Jupyter notebooks step-by-step:

* `notebooks/01_data_fetch_and_clean.ipynb`
* `notebooks/02_eda.ipynb`
* `notebooks/03_model_arima_lstm.ipynb`
* `notebooks/04_forecasting.ipynb`

---

## Results Summary

* **Data Cleaning:** Missing values handled, dates properly formatted, and consistent time series constructed.
* **EDA:** TSLA showed high volatility and non-stationarity in prices; returns were more stationary.
* **ARIMA/SARIMA:** Identified best seasonal parameters with grid search (e.g., SARIMA(0,1,1)x(0,1,1,12)) for Tesla price forecasting.
* **LSTM:** Deep learning model showed lower MAE and RMSE compared to ARIMA, indicating improved accuracy but higher complexity.
* **Evaluation Metrics:**

  * ARIMA MAE \~ 57.95, RMSE \~ 69.33
  * LSTM MAE \~ 9.92, RMSE \~ 13.77

---

## Future Directions

* Expand forecasting models with additional techniques (GARCH, Prophet).
* Integrate macroeconomic variables and sentiment data for enriched features.
* Implement portfolio optimization and risk budgeting modules.
* Automate the full data pipeline and enable real-time forecasting.
* Add backtesting of investment strategies using forecast results.

---

## Contact

**Dagmawi Ayenew**
Email: [ayenewdagmawi@gmail.com](mailto:ayenewdagmawi@gmail.com)

