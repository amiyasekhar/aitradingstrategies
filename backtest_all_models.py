#!/usr/bin/env python3
"""
backtest_all_models.py
Vectorised 30-day evaluation of every RF fold, in parallel.

Per-model result → results_30d/<model>.txt
Overall summary  → rf_30d_summary.csv
"""
from __future__ import annotations
import os, glob, textwrap, datetime as dt, joblib, pandas as pd, numpy as np
from joblib import Parallel, delayed
from env_minute import MinuteTradingEnv
from config import (HIST_DAYS, THRESH_UP, THRESH_DN,
                      PROBA_GAP, print_config, FEE, SLIPPAGE)
# ── project helpers -----------------------------------------------------------
from data_engineering       import fetch_history
from utils.threshold_search import load_threshold
# -----------------------------------------------------------------------------


# ─────────────── USER SETTINGS ────────────────────────────────────────────────
MODEL_GLOB  = "models/rf_fold*.joblib"
RESULT_DIR  = "results_30d"
START_DATE  = pd.Timestamp("2025-06-29", tz="UTC")   # inclusive
END_DATE    = pd.Timestamp("2025-07-29", tz="UTC")   # exclusive
N_CORES     = min(16, os.cpu_count() or 4)

# trading-cost assumptions (same as MinuteTradingEnv defaults)
# ──────────────────────────────────────────────────────────────────────────────


# ───────────────────────── helpers ────────────────────────────────────────────
def _prep_slice():
    span_days = (END_DATE - START_DATE).days + 1
    raw = fetch_history(span_days)

    # ---- Step 3: make index tz-aware & keep *all* columns, incl. "close"
    raw.index = (raw.index.tz_localize("UTC") if raw.index.tz is None
                 else raw.index.tz_convert("UTC"))
    m = (raw.index >= START_DATE) & (raw.index < END_DATE)
    feats  = raw.loc[m].copy()           # keep every feature column
    prices = feats["close"]
    return feats, prices, span_days


# ADD THIS NEW, FAST BACKTEST FUNCTION
def _vectorized_backtest(rf, df: pd.DataFrame, price: pd.Series,
                         up: float, dn: float, gap: float) -> dict:
    """
    A fast, vectorized backtest that accurately mirrors the cost
    logic from the MinuteTradingEnv (FEE + SLIPPAGE).
    """
    # --- Signal Generation ---
    X = df[rf.feature_names_in_]
    prob = rf.predict_proba(X)[:, 1]
    gap_m = np.abs(prob - .5) >= gap
    signal = np.where(prob > up,  1,
              np.where(prob < dn, -1, 0))
    signal = signal * gap_m

    # --- PnL Calculation ---
    pos = pd.Series(signal, index=df.index).replace(0, np.nan).ffill().fillna(0.).to_numpy()
    rets = price.pct_change().shift(-1).fillna(0.).to_numpy()

    # Accurately model costs from MinuteTradingEnv
    pos_change = np.abs(np.diff(np.concatenate(([0.], pos))))
    trade_cost = FEE + SLIPPAGE
    costs = pos_change * trade_cost

    pnl = pos * rets - costs
    equity = 1 + np.cumsum(pnl)

    # --- Metrics ---
    win_mask = pos != 0
    wins = ((pos > 0) & (rets > 0)) | ((pos < 0) & (rets < 0))
    win_rate = float(wins[win_mask].mean() * 100) if win_mask.any() else 0.
    sharpe = (pnl.mean() / pnl.std() * np.sqrt(252 * 6.5 * 60)) if pnl.std() > 0 else 0.0

    return {
        "total_return": (equity[-1] - 1) * 100,
        "equity_mult": float(equity[-1]),
        "sharpe": sharpe,
        "trades": int((pos_change > 0).sum()),
        "win_rate": win_rate,
    }


# REPLACE the old _run_one function WITH THIS ONE
def _run_one(model_path: str, feats: pd.DataFrame, prices: pd.Series,
             span_days: int):
    tag = os.path.splitext(os.path.basename(model_path))[0]
    rf = joblib.load(model_path)

    tuned = load_threshold("models", tag) or {}
    up = tuned.get("THRESH_UP", THRESH_UP)
    dn = tuned.get("THRESH_DN", THRESH_DN)
    gap = tuned.get("PROBA_GAP", PROBA_GAP)

    if any(c not in feats.columns for c in rf.feature_names_in_):
        missing = [c for c in rf.feature_names_in_ if c not in feats.columns]
        raise ValueError(f"{tag}: slice is missing columns {missing}")

    # Call the new fast backtester
    stats = _vectorized_backtest(rf, feats, prices, up, dn, gap)

    # friendly per-model TXT
    header = textwrap.dedent(f"""\
        # Model  : {tag}
        # Slice  : {START_DATE.date()} → {END_DATE.date()}  ({span_days} days)
        # Thresh : up={up:.2f}  dn={dn:.2f}  gap={gap:.2f}
    """)
    body = (
        f"Total return   : {stats['total_return']:+.2f}%\n"
        f"Equity multiple: {stats['equity_mult']:.2f}\n"
        f"Sharpe ratio   : {stats['sharpe']:.2f}\n"
        f"Trades / Win%  : {stats['trades']} | {stats['win_rate']:.2f}%\n"
    )
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, f"{tag}.txt"), "w") as fh:
        fh.write(header + body)

    return {"model": tag, **stats}


# ────────────────────────── main driver ───────────────────────────────────────
def main() -> None:
    feats, prices, span_days = _prep_slice()

    paths = sorted(glob.glob(MODEL_GLOB))
    print(f"▶  Evaluating {len(paths)} folds on 30-day slice using {N_CORES} core(s)…")

    rows = Parallel(n_jobs=N_CORES, backend="loky", verbose=0)(
        delayed(_run_one)(p, feats, prices, span_days) for p in paths
    )

    df = (pd.DataFrame(rows)
            .sort_values("total_return", ascending=False)
            .reset_index(drop=True))
    df.to_csv("rf_30d_summary.csv", index=False)

    print("\n✅  DONE – summary saved to rf_30d_summary.csv\n")
    print(df.head(15).to_string(index=False,
                                formatters={"total_return":"{:+.2f}%".format,
                                            "sharpe":"{:.2f}".format}))


if __name__ == "__main__":
    main()