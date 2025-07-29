# download_data.py
from data_engineering import fetch_history

print("Starting download of historical data for training and testing...")

# Download data from Aug 2023 to July 2025 to cover all periods.
df = fetch_history(start_date="2023-08-01", end_date="2025-07-30")

df.to_parquet("full_history.parquet")
print(f"✅ Data saved to full_history.parquet ({len(df)} rows)")