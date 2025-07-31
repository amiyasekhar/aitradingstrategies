# download_data.py
from data_engineering import fetch_history

print("Starting one-time download of full historical data...")
df = fetch_history(start_date="2020-01-01", end_date="2025-07-30")
df.to_parquet("full_history.parquet")
print(f"✅ Data saved to full_history.parquet ({len(df)} rows)")