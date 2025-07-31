#!/usr/bin/env python3
"""
run_backtest.py – back-tests one model.
* RF models can be evaluated in two flavours
    • default  : Gym loop (identical to before)
    • --fast   : fully-vectorised quick run  (≈100-150× faster)
* New flags
    --start YYYY-MM-DD
    --end   YYYY-MM-DD
    --days  N                # “last N calendar days” convenience
"""
from __future__ import annotations
import os, sys, argparse, json, datetime as dt, joblib
from pathlib import Path
from config import (HIST_DAYS, PAIR, MODEL_DIR, THRESH_UP, THRESH_DN,
                    PROBA_GAP, print_config, FEE, SLIPPAGE)
import numpy as np, pandas as pd

from data_engineering import fetch_history
from env_minute      import MinuteTradingEnv
from utils.metrics   import (bootstrap_equity, compute_sharpe_ratio,
                             compute_signal_classification_metrics)
from utils.threshold_search import load_threshold


# ────────────────────────────────────────────────────────────── helper ──
def _find_model(p: str | None) -> str:
    """Return a concrete model pathname (latest *.joblib if nothing given)."""
    if p and os.path.isfile(p):
        return p
    for fn in sorted(os.listdir(MODEL_DIR), reverse=True):
        if fn.lower().endswith((".joblib", ".pkl", ".zip")):
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


# ─────────────────────────────────────────────── vectorised RF back-test ──
def _quick_backtest_rf(rf, df: pd.DataFrame, up: float, dn: float, gap: float) -> dict:
    """Vectorised backtest where positions are held only when a signal is active."""
    price = df["close"]
    X = df[rf.feature_names_in_]
    prob  = rf.predict_proba(X)[:, 1]
    gap_m = np.abs(prob - .5) >= gap
    
    desired_position = np.where(prob > up, 1, np.where(prob < dn, -1, 0))
    desired_position = desired_position * gap_m
    position = pd.Series(desired_position, index=df.index).shift(1).fillna(0)

    rets = price.pct_change().fillna(0.)
    pnl = position * rets
    trades = position.diff().abs()
    costs = trades * (FEE + SLIPPAGE)
    net_pnl = pnl - costs
    
    equity = 1 + np.cumsum(net_pnl)
    equity_s = pd.Series(equity, index=df.index)

    in_trade = position != 0
    # --- FIXED: Add .astype(bool) to prevent FutureWarning ---
    trade_starts = in_trade & ~in_trade.shift(1).fillna(False).astype(bool)
    trade_ids = trade_starts.cumsum()
    trade_pnl = net_pnl[in_trade].groupby(trade_ids[in_trade]).sum()
    
    successful_trades = int((trade_pnl > 0).sum())
    failed_trades = int((trade_pnl <= 0).sum())
    
    acc, prec, rec, _ = compute_signal_classification_metrics(position.to_numpy(), price)

    return {
        "total_return": (equity.iloc[-1] - 1) * 100,
        "equity_mult" : float(equity.iloc[-1]),
        "sharpe"      : (net_pnl.mean() / net_pnl.std() * np.sqrt(252*6.5*60)) if net_pnl.std() else 0.,
        "successful_trades": successful_trades,
        "failed_trades": failed_trades,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "equity_s"    : equity_s,
    }


# ──────────────────────────────────────────────────────────── main ──
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="path/to/model.joblib | .zip")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--days", type=int)
    ap.add_argument("--fast", action="store_true", help="vectorised RF path (≈100× faster)")
    args = ap.parse_args()

    print_config()

    # 1) Load the full historical data from the local file
    print("Loading full historical data from local file...")
    try:
        raw_full = pd.read_parquet("full_history.parquet")
    except FileNotFoundError:
        print("❌ Error: full_history.parquet not found.")
        print("Please run the download_data.py script first.")
        return

    # 2) Slice the data to the requested window
    raw = _slice(raw_full,
                 dt.datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None,
                 dt.datetime.strptime(args.end,   "%Y-%m-%d").date() if args.end   else None,
                 args.days)

    print(f"Testing on data slice: {len(raw)} rows from {raw.index.min()} to {raw.index.max()}")
    if len(raw) == 0:
        print("❌ Error: No data found for the specified date range. Exiting.")
        return

    # 3) Load model / thresholds
    mdl_path = _find_model(args.model)
    is_rl = mdl_path.lower().endswith(".zip")
    print(f"🔍  Loading {'DQN' if is_rl else 'RandomForest'} model from “{mdl_path}”…")

    rf = None
    if not is_rl:
        rf = joblib.load(mdl_path)
        tag = Path(mdl_path).stem
        tuned = load_threshold(MODEL_DIR, tag) or {}
        up  = tuned.get("THRESH_UP", THRESH_UP)
        dn  = tuned.get("THRESH_DN", THRESH_DN)
        gap = tuned.get("PROBA_GAP", PROBA_GAP)
        print(f"   → thresholds   up={up}  dn={dn}  gap={gap}")

    # --- Path 1: Fast, Vectorized Backtest for RF models ---
    if not is_rl and args.fast:
        print("🚀 Running fast, vectorized backtest...")
        res = _quick_backtest_rf(rf, raw, up, dn, gap)
        eq  = res["equity_s"]
        start, end = eq.index[[0, -1]]
        btc = (raw["close"].iloc[-1] / raw["close"].iloc[0] - 1) * 100

        print(f"\n🕒  Testing period: {start} → {end}")
        print(f"💰  Total return: {res['total_return']:+.2f}%")
        print(f"📉  Buy-and-hold BTC: {btc:+.2f}%")
        print(f"📈  Successful Trades: {res['successful_trades']} | Failed Trades: {res['failed_trades']}")
        print(f"📊  Sharpe: {res['sharpe']:.2f}")
        print(f"🔍  Accuracy: {res['accuracy']:.2%} | Precision: {res['precision']:.2%} | Recall: {res['recall']:.2%}")

        em = eq.iloc[-1]
        mu, sig, var95 = bootstrap_equity(eq)
        print(f"📈  Equity multiple: {em:.2f} (boot μ {mu:.2f} σ {sig:.2f} "
              f"VaR95 {var95:.2f})")
        return

    # --- Path 2: Gym Loop (for RL models or normal RF backtest) ---
    if is_rl and args.fast:
        print("⚠️  --fast ignored for RL models – falling back to Gym.")

    print("🐌 Running detailed, step-by-step backtest using Gym environment...")

    if is_rl:
        from stable_baselines3 import DQN
        agent = DQN.load(mdl_path)

    env   = MinuteTradingEnv(raw)
    obs,_ = env.reset()
    equity, acts, ts, prices = [], [], [], []

    while True:
        if is_rl:
            action = int(agent.predict(obs, deterministic=True)[0])
        else:
            features = raw[rf.feature_names_in_].iloc[env.pointer - 1].values.reshape(1, -1)
            p = rf.predict_proba(features)[0, 1]
            gp = abs(p - 0.5)
            action = 1 if p > up and gp >= gap else 2 if p < dn and gp >= gap else 0

        obs, _, done, _, _ = env.step(action)
        equity.append(env.equity)
        acts.append(action)
        ts.append(raw.index[env.pointer - 1])
        prices.append(raw["close"].iloc[env.pointer - 1])

        if done:
            break

    equity_s = pd.Series(equity, index=ts)
    start, end = equity_s.index[[0, -1]]
    btc = (raw["close"].iloc[-1] / raw["close"].iloc[0] - 1) * 100

    print(f"\n🕒  Testing period: {start} → {end}")
    print(f"💰  Total return: {(equity_s.iloc[-1] - 1) * 100:+.2f}%")
    print(f"📉  Buy-and-hold BTC: {btc:+.2f}%")

    st = env.get_performance_stats()
    print(f"📉  Trades taken: {st['Total Trades']}  |  Win rate: {st['Win Rate (%)']} %")
    print(f"📊  Sharpe: {compute_sharpe_ratio(equity_s):.2f}")

    acc, mp, mr, _ = compute_signal_classification_metrics(
        np.array(acts), pd.Series(prices, index=ts))
    print(f"🔍  Accuracy: {acc:.2%}  |  Macro-P {mp:.2%}  R {mr:.2%}")

    em = equity_s.iloc[-1]
    mu, sig, var95 = bootstrap_equity(equity_s)
    print(f"📈  Equity multiple: {em:.2f} (boot μ {mu:.2f} σ {sig:.2f} VaR95 {var95:.2f})")


if __name__ == "__main__":
    main()