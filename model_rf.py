"""
Walk-forward and single-model training for Random Forest.
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

    rng = np.random.default_rng(42)
    idx_0, idx_1 = np.where(y_all == 0)[0], np.where(y_all == 1)[0]
    n = min(len(idx_0), len(idx_1))
    keep = np.concatenate([rng.choice(idx_0, n, replace=False), rng.choice(idx_1, n, replace=False)])
    X_b, y_b = X_all.iloc[keep], y_all[keep]
    print(f"Training single model: balanced counts 0→{(y_b == 0).sum()}, 1→{(y_b == 1).sum()}")

    edge = np.abs(df.loc[X_b.index, "next_ret"]) - COST_RT
    sample_weight = np.maximum(edge, 1e-6)

    rf = RandomForestClassifier(**FOREST_PARAMS)
    rf.fit(X_b, y_b, sample_weight=sample_weight)
    
    mdl_path = os.path.join(MODEL_DIR, "rf_master.joblib")
    joblib.dump(rf, mdl_path)
    print(f"\n✅ Master model saved to {mdl_path}")
    
    return rf


# ────────────────────────────────────────────────────────────────────
def train_walkforward(df: pd.DataFrame) -> RandomForestClassifier:
    """Performs walk-forward validation, creating multiple model folds."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_all = df.select_dtypes(include=[np.number]).drop(columns=["next_ret", "y"])
    y_all = df["y"].values
    final_model = None
    results: list[dict] = []

    # --- FIX: Convert days from config into minutes (rows) ---
    MINUTES_PER_DAY = 24 * 60
    TRAIN_WINDOW_MINUTES = HIST_DAYS * MINUTES_PER_DAY
    STEP_MINUTES = WALKFORWARD * MINUTES_PER_DAY

    fold = 0
    while True:
        # Use the new minute-based variables
        tr0 = fold * STEP_MINUTES
        tr1 = tr0 + TRAIN_WINDOW_MINUTES
        te1 = tr1 + STEP_MINUTES

        if te1 > len(df):
            break

        X_tr, y_tr = X_all.iloc[tr0:tr1], y_all[tr0:tr1]
        X_te, y_te = X_all.iloc[tr1:te1], y_all[tr1:te1]

        rng = np.random.default_rng(42 + fold)
        idx_0, idx_1 = np.where(y_tr == 0)[0], np.where(y_tr == 1)[0]
        n = min(len(idx_0), len(idx_1))
        keep = np.concatenate([rng.choice(idx_0, n, replace=False), rng.choice(idx_1, n, replace=False)])
        X_b, y_b = X_tr.iloc[keep], y_tr[keep]
        print(f"Fold {fold}: balanced counts 0→{(y_b == 0).sum()}, 1→{(y_b == 1).sum()}")

        edge = np.abs(df.loc[X_b.index, "next_ret"]) - COST_RT
        sample_weight = np.maximum(edge, 1e-6)

        rf = RandomForestClassifier(**FOREST_PARAMS)
        rf.fit(X_b, y_b, sample_weight=sample_weight)

        p_val = rf.predict_proba(X_te)[:, 1]
        best_t, best_eq = best_thresholds(p_val, df["close"].iloc[tr1:te1], COST_RT)
        save_threshold(f"rf_fold{fold}", best_t, MODEL_DIR)

        if np.isnan(p_val).any():
            p_val = np.nan_to_num(p_val, nan=0.5)
        auc = (np.nan if len(np.unique(y_te)) < 2 else roc_auc_score(y_te, p_val))

        results.append(dict(fold=fold, auc=auc, eq=best_eq, thr_up=best_t[0], gap=best_t[2]))

        mdl_path = os.path.join(MODEL_DIR, f"rf_fold{fold}.joblib")
        joblib.dump(rf, mdl_path)
        final_model = rf
        fold += 1

    summary = pd.DataFrame(results)
    summary_path = os.path.join(MODEL_DIR, "rf_folds.csv")
    summary.to_csv(summary_path, index=False)

    if not summary.empty:
        best_row  = summary.loc[summary["eq"].idxmax()]
        best_fold = int(best_row["fold"])
        best_src  = os.path.join(MODEL_DIR, f"rf_fold{best_fold}.joblib")
        best_link = os.path.join(MODEL_DIR, "rf_best.joblib")
        try:
            if os.path.lexists(best_link):
                os.remove(best_link)
            os.symlink(os.path.basename(best_src), best_link)
            print(f"\n⭐ Best fold {best_fold} (eq {best_row['eq']:.2f}) → rf_best.joblib")
        except Exception as e:
            print(f"Could not create symlink: {e}")

    print("\nWalk-forward results:")
    print(summary.to_string(index=False))
    return final_model