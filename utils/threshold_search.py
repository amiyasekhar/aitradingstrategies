# utils/threshold_search.py
import json, itertools
from pathlib import Path
import numpy as np
import pandas as pd

FEE_RT = 0.0022          # 22 bp round-trip

# ---------------------------------------------------------------------
def simulate_equity(actions: np.ndarray, prices: pd.Series,
                    fee_rt: float = FEE_RT) -> float:
    """Very light simulator: closes & re-opens with single-side fee."""
    equity, pos = 1.0, 0
    for act, price, nxt in zip(actions, prices, prices.shift(-1)):
        if np.isnan(nxt): break
        if act == 1 and pos <= 0:          # go long / flip
            equity *= 1 - fee_rt / 2
            pos = 1; entry = price
        elif act == 2 and pos >= 0:        # go short / flip
            equity *= 1 - fee_rt / 2
            pos = -1; entry = price
        elif act == 0 and pos != 0:        # flat exit
            equity *= 1 - fee_rt / 2
            pos = 0
        # unrealised PnL on minute bar
        if pos == 1:
            equity *= (nxt / price)
        elif pos == -1:
            equity *= (price / nxt)
    return equity

# ---------------------------------------------------------------------
def best_thresholds(proba: np.ndarray,
                    prices: pd.Series,
                    fee_rt: float = FEE_RT):
    """Return (th_up, th_dn, gap) that maximises ending equity."""
    grid_up   = np.arange(0.55, 0.71, 0.02)
    grid_gap  = (0.05, 0.08, 0.10, 0.12)
    best_eq, best_tuple = -np.inf, (0.6, 0.4, 0.05)

    for up, gap in itertools.product(grid_up, grid_gap):
        dn  = 1 - up
        a   = np.where((proba > up) & (np.abs(proba-0.5) >= gap), 1,
             np.where((proba < dn) & (np.abs(proba-0.5) >= gap), 2, 0))
        eq  = simulate_equity(a.astype(int), prices, fee_rt)
        if eq > best_eq:
            best_eq, best_tuple = eq, (round(up,2), round(dn,2), round(gap,2))
    return best_tuple, best_eq

# convenience persist helpers ----------------------------------------
def save_threshold(tag: str, t: tuple[float,float,float], model_dir: str):
    path = Path(model_dir) / "rf_thresholds.json"
    db   = {}
    if path.exists():
        db = json.loads(path.read_text())
    db[tag] = {"THRESH_UP": t[0], "THRESH_DN": t[1], "PROBA_GAP": t[2]}
    path.write_text(json.dumps(db, indent=2))

def load_threshold(model_dir: str, tag: str | None = None):
    """
    Return the tuned threshold triple for a given fold tag
    (e.g. 'rf_fold562'). If tag is None or not found, fall back to
    the most-recent entry.
    """
    path = Path(model_dir) / "rf_thresholds.json"
    if not path.exists():
        return None
    db = json.loads(path.read_text())
    if tag and tag in db:
        return db[tag]
    # fallback: most-recent entry
    return list(db.values())[-1]