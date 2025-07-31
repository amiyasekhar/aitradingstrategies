# run_rf_training.py
import pandas as pd
from model_rf import make_labels, train_walkforward

print("Loading full historical data from local file...")
try:
    raw_full = pd.read_parquet("full_history.parquet")
except FileNotFoundError:
    print("❌ Error: full_history.parquet not found.")
    print("Please run the download_data.py script first.")
    exit()

# Define the training period and slice the data
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = "2025-01-31"
start_ts = pd.to_datetime(TRAIN_START_DATE, utc=True)
end_ts = pd.to_datetime(TRAIN_END_DATE, utc=True)

training_data = raw_full.loc[start_ts:end_ts]
print(f"Starting walk-forward training on: {len(training_data)} rows from {training_data.index.min()} to {training_data.index.max()}")

df_labeled = make_labels(training_data)
train_walkforward(df_labeled)

print("\n✅ Walk-forward training complete.")