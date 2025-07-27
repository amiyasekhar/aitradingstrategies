#!/usr/bin/env python3
# run_backtest.py

import os, sys, argparse, joblib
import numpy as np, pandas as pd

from data_engineering import fetch_history
from env_minute import MinuteTradingEnv
from utils.metrics import (
    bootstrap_equity,
    compute_sharpe_ratio,
    compute_signal_classification_metrics,
)
from config import (
    HIST_DAYS,
    PAIR,
    MODEL_DIR,
    THRESH_UP,
    THRESH_DN,
    PROBA_GAP,
    print_config,
)


def _find_model(p: str | None) -> str:
    if p and os.path.isfile(p):
        return p
    for fn in os.listdir(MODEL_DIR):
        if fn.lower().endswith((".zip", ".pkl", ".joblib")):
            return os.path.join(MODEL_DIR, fn)
    sys.exit("❌ No model file found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()

    print_config()
    print(f"📊 Fetching {HIST_DAYS} days of {PAIR} 1m data…")
    raw = fetch_history(HIST_DAYS)

    mdl_path = _find_model(args.model)
    is_rl = mdl_path.lower().endswith(".zip")
    print(f"🔍 Loading {'DQN' if is_rl else 'RandomForest'} model from '{mdl_path}'…")

    if is_rl:
        from stable_baselines3 import DQN

        agent = DQN.load(mdl_path)
    else:
        rf = joblib.load(mdl_path)

    env = MinuteTradingEnv(raw)
    obs, _ = env.reset()

    equity, acts, ts, prices = [], [], [], []

    while True:
        if is_rl:
            action, _ = agent.predict(obs, deterministic=True)
            action = int(action)
        else:
            p = rf.predict_proba(raw.iloc[env.pointer - 1 : env.pointer])[0, 1]
            gap = abs(p - 0.5)
            action = (
                1
                if p > THRESH_UP and gap >= PROBA_GAP
                else 2
                if p < THRESH_DN and gap >= PROBA_GAP
                else 0
            )

        obs, _, done, _, _ = env.step(action)
        equity.append(env.equity)
        acts.append(action)
        ts.append(raw.index[env.pointer - 1])
        prices.append(raw["close"].iloc[env.pointer - 1])
        if done:
            break

    equity_s = pd.Series(equity, index=ts)
    start_ts, end_ts = equity_s.index[0], equity_s.index[-1]
    print(f"\n🕒 Testing period: {start_ts} → {end_ts}")

    ret_pct = (equity_s.iloc[-1] - 1) * 100
    print(f"💰 Total return: {ret_pct:+.2f}%")

    # ── NEW: passive BTC benchmark ───────────────────────────────────
    btc_ret = (raw["close"].iloc[-1] / raw["close"].iloc[0] - 1) * 100
    print(f"📉 Buy-and-hold BTC: {btc_ret:+.2f}% over same period")

    stats = env.get_performance_stats()
    print(f"📉 Trades taken: {stats['Total Trades']}")
    print(f"🏆 Win rate:     {stats['Win Rate (%)']} %")
    print(f"📊 Sharpe ratio (ann.): {compute_sharpe_ratio(equity_s):.2f}")

    acc, mp, mr, per = compute_signal_classification_metrics(
        np.array(acts), pd.Series(prices, index=ts)
    )
    print(f"🔍 Classification accuracy : {acc:.2%}")
    print(f"🔍 Macro-avg precision      : {mp:.2%}")
    print(f"🔍 Macro-avg recall         : {mr:.2%}")
    print("🔍 Precision / Recall by class:")
    for lbl, (prec, rec) in per.items():
        nm = {1: "Long", -1: "Short", 0: "Hold"}[lbl]
        print(f"    • {nm:5s} — precision: {prec:.2%}, recall: {rec:.2%}")

    em = equity_s.iloc[-1]
    mu, sigma, var95 = bootstrap_equity(equity_s)
    print(f"\n📈 Equity multiple: {em:.2f}")
    print(f"   Bootstrapped mean: {mu:.2f}, σ: {sigma:.2f}, VaR95: {var95:.2f}")


if __name__ == "__main__":
    main()