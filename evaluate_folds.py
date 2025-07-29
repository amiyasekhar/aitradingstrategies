#!/usr/bin/env python3
"""
evaluate_folds.py   –   fast, parallel evaluation of every RF fold

• Vectorised back-test: no Gym loop ➜ runs in seconds–minutes
• Uses per-fold .th.json thresholds when present,
  otherwise falls back to THRESH_UP / THRESH_DN / PROBA_GAP
• Writes fold_results.txt and (over-)writes models/rf_best.joblib
"""

from __future__ import annotations
import json, os, joblib, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, UTC
from joblib import Parallel, delayed

# ── project imports ──────────────────────────────────────────────
from config            import (HIST_DAYS, THRESH_UP, THRESH_DN,
                               PROBA_GAP, print_config)
from data_engineering  import fetch_history

# ----------------------------------------------------------------
MODEL_DIR   = Path("models")
LOG_FILE    = Path("fold_results.txt")
N_JOBS      = 8                   # CPU cores for joblib
RANK_BY     = "total_return"      # metric to maximise

# ── ultra-fast RF back-test (vectorised) ─────────────────────────
def quick_backtest_rf(rf, X: pd.DataFrame, price: pd.Series,
                      up: float, dn: float, gap: float) -> dict:
    proba  = rf.predict_proba(X)[:, 1]
    gap_ok = np.abs(proba - 0.5) >= gap
    acts   = np.where(proba > up,  1,
              np.where(proba < dn,  2, 0))
    acts  *= gap_ok

    rets   = price.pct_change().shift(-1).fillna(0.0).to_numpy()
    pnl    = np.where(acts == 1,  rets,
              np.where(acts == 2, -rets, 0.0))
    equity = 1 + np.cumsum(pnl)

    return {
        "total_return": (equity[-1] - 1) * 100,
        "equity_mult" : float(equity[-1]),
        "sharpe"      : (pnl.mean() / pnl.std() * np.sqrt(252*6.5*60)
                         if pnl.std() else 0.0),
        "trades"      : int((acts != 0).sum()),
        "win_rate"    : float(((acts == 1) & (rets > 0) |
                               (acts == 2) & (rets < 0)).mean() * 100)
    }

# ── worker for one fold ─────────────────────────────────────────
def evaluate_fold(joblib_path: Path, X: pd.DataFrame,
                  price: pd.Series) -> tuple[str, dict]:
    rf = joblib.load(joblib_path)

    th_file = joblib_path.with_suffix(".th.json")
    if th_file.exists():
        th   = json.loads(th_file.read_text())
        up   = th.get("THRESH_UP",  THRESH_UP)
        dn   = th.get("THRESH_DN",  THRESH_DN)
        gap  = th.get("PROBA_GAP", PROBA_GAP)
    else:
        up, dn, gap = THRESH_UP, THRESH_DN, PROBA_GAP

    stats = quick_backtest_rf(rf, X, price, up, dn, gap)
    return joblib_path.stem, stats

# ── main ─────────────────────────────────────────────────────────
def main() -> None:
    print_config()
    folds = sorted(MODEL_DIR.glob("rf_fold*.joblib"))
    if not folds:
        print("❌  No Random-Forest fold files found under ./models")
        return

    # one 30-day data frame for all folds
    raw   = fetch_history(HIST_DAYS)
    price = raw["close"]
    X     = raw.reset_index(drop=True)

    print(f"🚀  Evaluating {len(folds)} folds on {N_JOBS} CPU cores …")
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(evaluate_fold)(p, X, price) for p in folds
    )

    # rank and print
    results  = sorted(results, key=lambda x: x[1][RANK_BY], reverse=True)
    now_iso  = datetime.now(UTC).isoformat(timespec="seconds")
    header   = (f"# Fold evaluation {now_iso}\n"
                f"# {'fold':<12}{'ret%':>8}{'shr':>8}{'wins%':>8}{'eq×':>8}\n")
    LOG_FILE.write_text(header)
    print(header, end="")

    for fd, st in results:
        line = f"{fd:<12}{st['total_return']:8.2f}{st['sharpe']:8.2f}" \
               f"{st['win_rate']:8.2f}{st['equity_mult']:8.2f}\n"
        LOG_FILE.write_text(LOG_FILE.read_text() + line)
        print(line, end="")

    best_fold, best_stats = results[0]
    print(f"\n🏆  Best fold = {best_fold}  "
          f"({RANK_BY} = {best_stats[RANK_BY]:.2f})")

    # always overwrite the symlink
    best_link = MODEL_DIR / "rf_best.joblib"
    try:
        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()
        best_link.symlink_to(f"{best_fold}.joblib")
        print(f"→  models/rf_best.joblib → {best_fold}.joblib")
    except OSError as e:
        print(f"⚠️  Could not create symlink: {e}")

if __name__ == "__main__":
    main()