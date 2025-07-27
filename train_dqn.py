#!/usr/bin/env python3
# train_dqn.py

# ─── MONKEY-PATCH FOR PANDAS_TA ─────────────────────────────────────
import pandas as pd
pd.Series.append = pd.Series._append   # restore the old .append used by pandas_ta

# ─── IMPORTS ─────────────────────────────────────────────────────────
import joblib
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from data_engineering import fetch_history, add_indicators
from model_rf         import make_labels, train_walkforward
from env_minute       import MinuteTradingEnv
from config           import HIST_DAYS

if __name__ == "__main__":
    # 1) Fetch historical data
    raw = fetch_history(HIST_DAYS)

    # 2) Compute technical indicators (fills NaNs)
    df = add_indicators(raw)

    # 3) Create labels DataFrame
    df_labeled = make_labels(df)

    # 4) Train Random Forest with walk-forward CV
    rf = train_walkforward(df_labeled)

    # 5) Prepare DataFrame for RL:
    df_labeled["label"] = df_labeled["y"]
    df_rl = df_labeled.drop(columns=["y", "next_ret"])

    # 6) Create Gym environment factory
    def make_env():
        return MinuteTradingEnv(df_rl)

    vec_env = DummyVecEnv([make_env])

    # 7) Train DQN agent
    dqn = DQN(
        policy="MlpPolicy",
        env=vec_env,
        buffer_size=50_000,
        learning_rate=1e-4,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.1,
        verbose=1,
    )
    dqn.learn(total_timesteps=1_500_000)

    # 8) Save models
    dqn.save("models/dqn_minute")
    joblib.dump(rf, "models/rf_minute.pkl")