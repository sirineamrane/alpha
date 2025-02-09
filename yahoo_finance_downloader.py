import yfinance as yf
import pandas as pd

# Define the stock symbol and time range
symbol = "SPY"  # Change this to the stock, ETF, or crypto you want (e.g., "AAPL", "BTC-USD")
start_date = "2010-01-01"
end_date = "2024-02-09"

# Download historical data
print(f"📥 Downloading {symbol} data from Yahoo Finance...")
data = yf.download(symbol, start=start_date, end=end_date)

# Keep only the adjusted close price
data = data[["Adj Close"]]
data.rename(columns={"Adj Close": "Price"}, inplace=True)

# Calculate daily returns
data["Return"] = data["Price"].pct_change()
data.dropna(inplace=True)

# Save to CSV
csv_filename = f"{symbol}_returns.csv"
data.to_csv(csv_filename, index=True, header=False)

print(f"✅ Data saved as {csv_filename}. Ready for use in the detection script.")
