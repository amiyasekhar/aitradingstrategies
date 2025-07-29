#!/usr/bin/env python3
# train_dqn.py

# ─── MONKEY-PATCH FOR PANDAS_TA ─────────────────────────────────────
import pandas as pd
pd.Series.append = pd.Series._append   # restore the old .append used by pandas_ta

# ─── IMPORTS ────────────────────────────────────────────────────────
import joblib
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from data_engineering import add_indicators
from model_rf         import make_labels, train_single_model
from env_minute       import MinuteTradingEnv

if __name__ == "__main__":
    # 1) Load the full historical data from the local file
    print("Loading full historical data from local file...")
    try:
        raw_full = pd.read_parquet("full_history.parquet")
    except FileNotFoundError:
        print("❌ Error: full_history.parquet not found.")
        print("Please run the download_data.py script first.")
        exit()

    # 2) Define the training period and slice the data
    TRAIN_START_DATE = "2020-01-01"
    TRAIN_END_DATE = "2025-01-31"
    start_ts = pd.to_datetime(TRAIN_START_DATE, utc=True)
    end_ts = pd.to_datetime(TRAIN_END_DATE, utc=True)
    
    raw = raw_full.loc[start_ts:end_ts]
    print(f"Using training slice: {len(raw)} rows from {raw.index.min()} to {raw.index.max()}")

    # 3) Compute technical indicators on the training slice
    df = add_indicators(raw)

    # 4) Create labels DataFrame
    df_labeled = make_labels(df)

    # 5) Train a single Random Forest master model
    rf = train_single_model(df_labeled)

    # 6) Prepare DataFrame for RL
    df_rl = df_labeled.drop(columns=["y", "next_ret"])

    # 7) Create Gym environment factory
    def make_env():
        return MinuteTradingEnv(df_rl)

    vec_env = DummyVecEnv([make_env])

    # 8) Train DQN agent
    dqn = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        exploration_initial_eps=0.30,
        exploration_final_eps=0.02,
        exploration_fraction=0.80,
        gamma=0.99,
        target_update_interval=500,
        verbose=1,
    )

    # --- Live debug callbacks ---
    class RewardProbe(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 5_000 == 0:
                last_rew = float(self.locals["rewards"][0])
                print(f"step {self.num_timesteps:,}  last_reward = {last_rew:+.5f}")
            return True

    class EpsilonTracker(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 100_000 == 0:
                eps = self.model.exploration_rate
                print(f"step {self.num_timesteps:,}  ε = {eps:.3f}")
            return True

    dqn.learn(
        total_timesteps=1_500_000,
        callback=[RewardProbe(), EpsilonTracker()],
    )

    # 9) Save models
    dqn.save("models/dqn_master")
    # The RF model is already saved by train_single_model
    print("✅ DQN master model saved to models/dqn_master")