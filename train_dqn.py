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

    # 5) Prepare DataFrame for RL (features only, 27 cols)
    df_rl = df_labeled.drop(columns=["y", "next_ret"])

    # 6) Create Gym environment factory
    def make_env():
        return MinuteTradingEnv(df_rl)

    vec_env = DummyVecEnv([make_env])

    # sanity-check feature shape
    print("🔍 Env feature shape =", vec_env.observation_space.shape)   # expect (60, 27)

    # 7) Train DQN agent
    dqn = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        exploration_initial_eps=0.30,
        exploration_final_eps=0.02,
        exploration_fraction=0.80,          # slower decay
        gamma=0.99,
        target_update_interval=500,
        verbose=1,
    )

    # ── live debug callbacks ────────────────────────────────────────
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

    # 8) Save models
    dqn.save("models/dqn_minute")
    joblib.dump(rf, "models/rf_minute.pkl")