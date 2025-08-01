#!/usr/bin/env python3
"""
backtest_all_models.py
Vectorised evaluation for Random Forest Regressor models.
"""
from __future__ import annotations
import os, glob, pandas as pd, numpy as np
import joblib
from joblib import Parallel, delayed
from config import FEE, SLIPPAGE, MODEL_DIR
from utils.metrics import compute_signal_classification_metrics

MODEL_GLOB = "models/rf_fold*.joblib"

IN_SAMPLE_START = pd.Timestamp("2020-01-01", tz="UTC")
IN_SAMPLE_END = pd.Timestamp("2025-01-31", tz="UTC")
OUT_OF_SAMPLE_START = pd.Timestamp("2025-02-01", tz="UTC")
OUT_OF_SAMPLE_END = pd.Timestamp("2025-07-30", tz="UTC")

N_CORES = 4

def _prep_slices():
    try:
        raw_full = pd.read_parquet("full_history.parquet")
    except FileNotFoundError:
        print("❌ Error: full_history.parquet not found. Run download_data.py first.")
        exit()
    
    m_in = (raw_full.index >= IN_SAMPLE_START) & (raw_full.index < IN_SAMPLE_END)
    df_in = raw_full.loc[m_in].copy()

    m_out = (raw_full.index >= OUT_OF_SAMPLE_START) & (raw_full.index < OUT_OF_SAMPLE_END)
    df_out = raw_full.loc[m_out].copy()
    
    return df_in, df_out

def _vectorized_backtest(rf, df: pd.DataFrame) -> dict:
    """Runs a backtest for a regressor model."""
    if df.empty:
        return {
            "return": 0.0, "sharpe": 0.0, "successful_trades": 0, "failed_trades": 0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0
        }

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

    in_trade = position != 0
    trade_starts = in_trade & ~in_trade.shift(1).fillna(False).astype(bool)
    trade_ids = trade_starts.cumsum()
    trade_pnl = net_pnl[in_trade].groupby(trade_ids[in_trade]).sum()
    
    successful_trades = int((trade_pnl > 0).sum())
    failed_trades = int((trade_pnl <= 0).sum())
    
    # --- ADDED: Calculate classification metrics for the regressor's signals ---
    acc, prec, rec, _ = compute_signal_classification_metrics(position.to_numpy(), price)
    
    return {
        "return": (equity.iloc[-1] - 1) * 100,
        "sharpe": (net_pnl.mean() / net_pnl.std() * np.sqrt(252*6.5*60)) if net_pnl.std() else 0.0,
        "successful_trades": successful_trades,
        "failed_trades": failed_trades,
        "accuracy": acc * 100,
        "precision": prec * 100,
        "recall": rec * 100
    }

def _run_one(model_path: str, df_in: pd.DataFrame, df_out: pd.DataFrame):
    tag = os.path.splitext(os.path.basename(model_path))[0]
    rf = joblib.load(model_path)
    
    stats_in = _vectorized_backtest(rf, df_in)
    stats_out = _vectorized_backtest(rf, df_out)
    
    # --- UPDATED: Return all metrics for both in-sample and out-of-sample ---
    return {
        "model": tag,
        "in_return": stats_in["return"],
        "in_sharpe": stats_in["sharpe"],
        "in_successful": stats_in["successful_trades"],
        "in_failed": stats_in["failed_trades"],
        "in_accuracy": stats_in["accuracy"],
        "in_precision": stats_in["precision"],
        "in_recall": stats_in["recall"],
        "out_return": stats_out["return"],
        "out_sharpe": stats_out["sharpe"],
        "out_successful": stats_out["successful_trades"],
        "out_failed": stats_out["failed_trades"],
        "out_accuracy": stats_out["accuracy"],
        "out_precision": stats_out["precision"],
        "out_recall": stats_out["recall"],
    }

def main():
    df_in, df_out = _prep_slices()
    paths = sorted(glob.glob(MODEL_GLOB))
    print(f"▶  Evaluating {len(paths)} regressor folds using {N_CORES} core(s)…")
    
    rows = Parallel(n_jobs=N_CORES, backend="loky", verbose=5)(delayed(_run_one)(p, df_in, df_out) for p in paths)
    
    summary = pd.DataFrame([r for r in rows if r]).sort_values("out_return", ascending=False).reset_index(drop=True)
    
    print("\n✅  DONE – Combined Regressor Backtest Results\n")
    
    formatters={
        "in_return":"{:,.2f}%".format,
        "in_sharpe":"{:.2f}".format,
        "in_accuracy":"{:.2f}%".format,
        "in_precision":"{:.2f}%".format,
        "in_recall":"{:.2f}%".format,
        "out_return":"{:,.2f}%".format,
        "out_sharpe":"{:.2f}".format,
        "out_accuracy":"{:.2f}%".format,
        "out_precision":"{:.2f}%".format,
        "out_recall":"{:.2f}%".format,
    }
    
    print(summary.to_string(index=False, formatters=formatters))

if __name__ == "__main__":
    main()