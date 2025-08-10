import yfinance as yf
import os

def fetch_and_save(ticker, start, end, path):
    data = yf.download(ticker, start=start, end=end, progress=False)
    # Confirm columns
    print(f"{ticker} columns fetched: {list(data.columns)}")
    data.to_csv(path)
    print(f"{ticker} data saved to {path}")

def main():
    os.makedirs("../data/raw", exist_ok=True)
    start_date = "2015-07-01"
    end_date = "2025-07-31"
    
    fetch_and_save("TSLA", start_date, end_date, "../data/raw/tsla_raw.csv")
    fetch_and_save("BND",  start_date, end_date, "../data/raw/bnd_raw.csv")
    fetch_and_save("SPY",  start_date, end_date, "../data/raw/spy_raw.csv")

if __name__ == "__main__":
    main()
