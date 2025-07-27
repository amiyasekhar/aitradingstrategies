# env_minute.py
"""
Minute-bar Gymnasium environment.

2025-07-27
──────────
• Fees/slippage charged only on fills.
• Reward clipped to ±0.01 × REWARD_SCALE.
• Win counted only when a *closed* trade is profitable.
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
        self.action_space = spaces.Discrete(3)

        self.pointer = WINDOW
        self.position = 0          # −1, 0, +1
        self.equity = 1.0
        self.entry_equity = 1.0    # equity at last entry
        self.trades = 0
        self.wins = 0
        self.max_equity = 1.0

    # ── helpers ─────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        return self.df.iloc[self.pointer - WINDOW : self.pointer].values.astype(
            np.float32
        )

    def _price(self) -> float:
        return float(self.df.iloc[self.pointer - 1]["close"])

    # ── core step ───────────────────────────────────────────────────
    def step(self, action: int):
        assert self.action_space.contains(action)
        done = False
        trade_cost = 0.0
        opened_or_closed = False

        # decide if we change position
        if action == 1 and self.position <= 0:     # go long / flip
            trade_cost = FEE + SLIPPAGE
            if self.position != 0:
                trade_cost += FEE + SLIPPAGE
                opened_or_closed = True            # closing old short
            self.position = 1
            self.trades += 1
            self.entry_equity = self.equity

        elif action == 2 and self.position >= 0:   # go short / flip
            trade_cost = FEE + SLIPPAGE
            if self.position != 0:
                trade_cost += FEE + SLIPPAGE
                opened_or_closed = True            # closing old long
            self.position = -1
            self.trades += 1
            self.entry_equity = self.equity

        # move to next bar
        prev_price = self._price()
        self.pointer += 1
        if self.pointer >= len(self.df):
            done = True
        price = self._price()
        ret = (price - prev_price) / prev_price
        pnl = self.position * ret - trade_cost
        self.equity *= 1 + pnl
        self.max_equity = max(self.max_equity, self.equity)

        # count win only when we *close* a trade profitably
        if opened_or_closed and self.equity > self.entry_equity:
            self.wins += 1

        reward = np.clip(pnl, -0.01, 0.01) * REWARD_SCALE
        return self._obs(), reward, done, False, {}

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