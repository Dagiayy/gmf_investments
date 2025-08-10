import os
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_data, basic_stats, check_missing, fill_missing, scale_data

def main():
    raw_folder = "data/raw"
    processed_folder = "data/processed"
    output_folder = "output"  # New folder for plots
    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    assets = ['tsla', 'bnd', 'spy']
    dfs = {}

    # Load raw data and show basic stats + missing values
    for asset in assets:
        file_path = os.path.join(raw_folder, f"{asset}_raw.csv")
        df = load_data(file_path)
        dfs[asset] = df
        print(f"\nBasic stats for {asset.upper()}:\n", basic_stats(df))
        print(f"\nMissing values for {asset.upper()}:\n", check_missing(df))

    # Fill missing values (forward fill)
    for asset, df in dfs.items():
        dfs[asset] = fill_missing(df, method='ffill')

    # Scale TSLA data - only columns that exist
    cols_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    tsla_cols_available = [col for col in cols_to_scale if col in dfs['tsla'].columns]
    tsla_scaled, scaler = scale_data(dfs['tsla'], tsla_cols_available, method='minmax')

    # Save processed data
    for asset, df in dfs.items():
        processed_path = os.path.join(processed_folder, f"{asset}_processed.csv")
        df.to_csv(processed_path)
        print(f"Saved processed data for {asset.upper()} at {processed_path}")

    # Save scaled TSLA separately
    tsla_scaled_path = os.path.join(processed_folder, "tsla_scaled.csv")
    tsla_scaled.to_csv(tsla_scaled_path)
    print(f"Saved scaled TSLA data at {tsla_scaled_path}")

    # Plot Closing Price distributions and save images
    plt.figure(figsize=(15,4))
    for i, asset in enumerate(assets, 1):
        plt.subplot(1, 3, i)
        sns.histplot(dfs[asset]['Close'], kde=True)
        plt.title(f"{asset.upper()} Close Price Distribution")
    plt.tight_layout()

    # Save the figure
    plot_path = os.path.join(output_folder, "closing_price_distribution.png")
    plt.savefig(plot_path)
    print(f"Saved closing price distribution plot at {plot_path}")

    # Show the plot interactively as well
    plt.show()

if __name__ == "__main__":
    main()
