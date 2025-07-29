"""
Walk-forward Random-Forest with edge-weighted samples,
per-fold threshold optimisation, and auto-selection of
the best fold.

Outputs
-------
models/rf_foldN.joblib       : one model per fold
models/rf_thresholds.json    : tuned thresholds per fold
models/rf_folds.csv          : table of fold, auc, eq, thr_up, gap
models/rf_best.joblib        : symlink to the highest-eq fold
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from utils.threshold_search import best_thresholds, save_threshold
from config import HIST_DAYS, FOREST_PARAMS, WALKFORWARD, MODEL_DIR, FEE, SLIPPAGE

# Use a realistic round-trip cost from the config file
COST_RT = 2 * (FEE + SLIPPAGE)


# ────────────────────────────────────────────────────────────────────
def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["next_ret"] = df["close"].pct_change().shift(-1)
    df["y"] = (df["next_ret"] > 0).astype(int)
    return df.dropna().reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────
def train_single_model(df: pd.DataFrame) -> RandomForestClassifier:
    """Trains a single Random Forest model on the entire dataset."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_all = df.select_dtypes(include=[np.number]).drop(columns=["next_ret", "y"])
    y_all = df["y"].values

    # Balance classes across the entire dataset
    rng = np.random.default_rng(42)
    idx_0, idx_1 = np.where(y_all == 0)[0], np.where(y_all == 1)[0]
    n = min(len(idx_0), len(idx_1))
    keep = np.concatenate(
        [rng.choice(idx_0, n, replace=False),
         rng.choice(idx_1, n, replace=False)]
    )
    X_b, y_b = X_all.iloc[keep], y_all[keep]
    print(f"Training single model: balanced counts 0→{(y_b == 0).sum()}, 1→{(y_b == 1).sum()}")

    # Edge-weighted sample weight
    COST_RT = 2 * (FEE + SLIPPAGE)
    edge = np.abs(df.loc[X_b.index, "next_ret"]) - COST_RT
    sample_weight = np.maximum(edge, 1e-6)

    # Fit a single RF model
    rf = RandomForestClassifier(**FOREST_PARAMS)
    rf.fit(X_b, y_b, sample_weight=sample_weight)
    
    # Save the master model
    mdl_path = os.path.join(MODEL_DIR, "rf_master.joblib")
    joblib.dump(rf, mdl_path)
    print(f"\n✅ Master model saved to {mdl_path}")
    
    return rf

    # ── save fold summary & best-eq symlink ──────────────────────────
    summary = pd.DataFrame(results)
    summary_path = os.path.join(MODEL_DIR, "rf_folds.csv")
    summary.to_csv(summary_path, index=False)

    best_row  = summary.loc[summary["eq"].idxmax()]
    best_fold = int(best_row["fold"])
    best_src  = os.path.join(MODEL_DIR, f"rf_fold{best_fold}.joblib")
    best_link = os.path.join(MODEL_DIR, "rf_best.joblib")
    try:
        os.remove(best_link)
    except FileNotFoundError:
        pass
    os.symlink(best_src, best_link)
    print(f"\n⭐ Best fold {best_fold} (eq {best_row['eq']:.2f}) "
          f"→ rf_best.joblib")

    print("\nWalk-forward results:")
    print(summary.to_string(index=False))
    return final_model