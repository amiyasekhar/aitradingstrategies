#!/usr/bin/env python3
"""
run_backtest.py – back-tests one regressor model.
"""
from __future__ import annotations
import os, sys, argparse, datetime as dt, joblib
from pathlib import Path
from config import (HIST_DAYS, PAIR, MODEL_DIR, print_config, FEE, SLIPPAGE)
import numpy as np, pandas as pd

from data_engineering import fetch_history
from env_minute import MinuteTradingEnv
from utils.metrics import bootstrap_equity, compute_sharpe_ratio

def _find_model(p: str | None) -> str:
    """Return a concrete model pathname (latest *.joblib if nothing given)."""
    if p and os.path.isfile(p):
        return p
    for fn in sorted(os.listdir(MODEL_DIR), reverse=True):
        if fn.lower().endswith((".joblib", ".pkl")):
            return os.path.join(MODEL_DIR, fn)
    sys.exit("❌  No model file found in MODEL_DIR.")

def _slice(df: pd.DataFrame,
           start: dt.date | None,
           end:   dt.date | None,
           last_n: int | None) -> pd.DataFrame:
    """Return a date-slice OR 'last N days' slice of the raw minute data."""
    if last_n:
        cutoff = df.index[-1].date() - dt.timedelta(days=last_n)
        return df[df.index.date > cutoff]
    if start:
        df = df[df.index.date >= start]
    if end:
        df = df[df.index.date <  end]
    return df

def _quick_backtest_rf(rf, df: pd.DataFrame) -> dict:
    """Vectorised backtest for a regressor model."""
    PROFIT_THRESHOLD = FEE + SLIPPAGE

    price = df["close"]
    X = df[rf.feature_names_in_]
    
    predicted_returns = rf.predict(X)

    desired_position = np.where(predicted_returns > PROFIT_THRESHOLD, 1,
                         np.where(predicted_returns < -PROFIT_THRESHOLD, -1, 0))
    
    position = pd.Series(desired_position, index=df.index).shift(1).fillna(0)

    rets = price.pct_change().fillna(0.)
    pnl = position * rets
    trades = position.diff().abs()
    costs = trades * (FEE + SLIPPAGE)
    net_pnl = pnl - costs
    
    equity = 1 + np.cumsum(net_pnl)
    equity_s = pd.Series(equity, index=df.index)

    in_trade = position != 0
    trade_starts = in_trade & ~in_trade.shift(1).fillna(False).astype(bool)
    trade_ids = trade_starts.cumsum()
    trade_pnl = net_pnl[in_trade].groupby(trade_ids[in_trade]).sum()
    
    successful_trades = int((trade_pnl > 0).sum())
    failed_trades = int((trade_pnl <= 0).sum())

    return {
        "total_return": (equity.iloc[-1] - 1) * 100,
        "sharpe": (net_pnl.mean() / net_pnl.std() * np.sqrt(252*6.5*60)) if net_pnl.std() else 0.,
        "successful_trades": successful_trades,
        "failed_trades": failed_trades,
        "equity_s": equity_s,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="path/to/model.joblib")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--days", type=int)
    args = ap.parse_args()

    print_config()

    try:
        raw_full = pd.read_parquet("full_history.parquet")
    except FileNotFoundError:
        print("❌ Error: full_history.parquet not found. Please run download_data.py first.")
        return

    raw = _slice(raw_full,
                 dt.datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None,
                 dt.datetime.strptime(args.end,   "%Y-%m-%d").date() if args.end   else None,
                 args.days)

    print(f"Testing on data slice: {len(raw)} rows from {raw.index.min()} to {raw.index.max()}")
    if len(raw) == 0:
        print("❌ Error: No data found for the specified date range. Exiting.")
        return

    mdl_path = _find_model(args.model)
    print(f"🔍  Loading RandomForest Regressor model from “{mdl_path}”…")
    rf = joblib.load(mdl_path)

    print("🚀 Running fast, vectorized backtest...")
    res = _quick_backtest_rf(rf, raw)
    eq  = res["equity_s"]
    start, end = eq.index[[0, -1]]
    btc = (raw["close"].iloc[-1] / raw["close"].iloc[0] - 1) * 100

    print(f"\n🕒  Testing period: {start} → {end}")
    print(f"💰  Total return: {res['total_return']:+.2f}%")
    print(f"📉  Buy-and-hold BTC: {btc:+.2f}%")
    print(f"📈  Successful Trades: {res['successful_trades']} | Failed Trades: {res['failed_trades']}")
    print(f"📊  Sharpe: {res['sharpe']:.2f}")

    em = eq.iloc[-1]
    mu, sig, var95 = bootstrap_equity(eq)
    print(f"📈  Equity multiple: {em:.2f} (boot μ {mu:.2f} σ {sig:.2f} VaR95 {var95:.2f})")

if __name__ == "__main__":
    main()