# env_minute.py
"""
Minute-bar Gymnasium environment.

• Fees/slippage charged only on fills
• Reward clipped and includes a tiny holding penalty so the agent
  learns to exit unproductive positions.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from config import WINDOW, FEE, SLIPPAGE, REWARD_SCALE, DRAWDOWN_LIMIT


class MinuteTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)

        n_features = df.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(WINDOW, n_features), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)          # 0 Hold, 1 Long, 2 Short

        self.pointer = WINDOW
        self.position = 0          # −1 short, 0 flat, +1 long
        self.equity = 1.0
        self.entry_equity = 1.0
        self.trades = 0
        self.wins = 0
        self.max_equity = 1.0

    # ────────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        return self.df.iloc[self.pointer - WINDOW : self.pointer].values.astype(
            np.float32
        )

    def _price(self) -> float:
        return float(self.df.iloc[self.pointer - 1]["close"])

    # ────────────────────────────────────────────────────────────
    def step(self, action: int):
        assert self.action_space.contains(action)
        done = False
        trade_cost = 0.0
        opened_or_closed = False

        # ── position management ─────────────────────────────────
        if action == 1 and self.position <= 0:          # go / flip long
            trade_cost = FEE + SLIPPAGE
            if self.position != 0:                      # closing short
                trade_cost += FEE + SLIPPAGE
                opened_or_closed = True
            self.position = 1
            self.trades += 1
            self.entry_equity = self.equity

        elif action == 2 and self.position >= 0:        # go / flip short
            trade_cost = FEE + SLIPPAGE
            if self.position != 0:                      # closing long
                trade_cost += FEE + SLIPPAGE
                opened_or_closed = True
            self.position = -1
            self.trades += 1
            self.entry_equity = self.equity

        # ── advance one bar ─────────────────────────────────────
        prev_price = self._price()
        self.pointer += 1
        if self.pointer >= len(self.df):
            done = True
        price = self._price()

        ret = (price - prev_price) / prev_price
        pnl = self.position * ret - trade_cost

        # ── NEW: holding penalty (1 bp per minute when in market) ──
        holding_penalty = 0.000001 * abs(self.position)   # 0.01 bp per bar
        pnl -= holding_penalty

        self.equity *= 1 + pnl
        self.max_equity = max(self.max_equity, self.equity)

        if opened_or_closed and self.equity > self.entry_equity:
            self.wins += 1

        reward = np.clip(pnl, -0.01, 0.01) * REWARD_SCALE
        return self._obs(), reward, done, False, {}

    # ────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pointer = WINDOW
        self.position = 0
        self.equity = 1.0
        self.entry_equity = 1.0
        self.trades = 0
        self.wins = 0
        self.max_equity = 1.0
        return self._obs(), {}

    def get_performance_stats(self) -> dict:
        win_rate = 100.0 * self.wins / max(self.trades, 1)
        return {
            "Total Trades": self.trades,
            "Win Rate (%)": f"{win_rate:.2f}",
            "Final Equity": f"{self.equity:.4f}",
        }