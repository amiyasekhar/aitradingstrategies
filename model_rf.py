"""
Walk-forward and single-model training for Random Forest Regressor.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from config import HIST_DAYS, FOREST_PARAMS, WALKFORWARD, MODEL_DIR, FEE, SLIPPAGE

COST_RT = 2 * (FEE + SLIPPAGE)

def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the target variable (y) for the regression model.
    'y' will be the actual future return.
    """
    df = df.copy()
    horizon = 60 

    future_ret = df["close"].pct_change(periods=horizon).shift(-horizon)
    
    df["y"] = future_ret
    df["next_ret"] = future_ret

    return df.dropna().reset_index(drop=True)

def train_single_model(df: pd.DataFrame) -> RandomForestRegressor:
    """Trains a single Random Forest Regressor on the entire dataset."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_all = df.select_dtypes(include=[np.number]).drop(columns=["next_ret", "y"])
    y_all = df["y"].values

    rng = np.random.default_rng(42)
    keep_indices = rng.choice(len(X_all), size=min(len(X_all), 500_000), replace=False)
    X_b, y_b = X_all.iloc[keep_indices], y_all[keep_indices]
    
    print(f"Training single regressor on {len(X_b)} samples...")

    edge = np.abs(df.loc[X_b.index, "next_ret"]) - COST_RT
    sample_weight = np.maximum(edge, 1e-6)

    rf = RandomForestRegressor(**FOREST_PARAMS)
    rf.fit(X_b, y_b, sample_weight=sample_weight)
    
    mdl_path = os.path.join(MODEL_DIR, "rf_master.joblib")
    joblib.dump(rf, mdl_path)
    print(f"\n✅ Master regressor model saved to {mdl_path}")
    
    return rf

def train_walkforward(df: pd.DataFrame) -> RandomForestRegressor:
    """Performs walk-forward validation for the regressor."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_all = df.select_dtypes(include=[np.number]).drop(columns=["next_ret", "y"])
    y_all = df["y"].values
    final_model = None

    MINUTES_PER_DAY = 24 * 60
    TRAIN_WINDOW_MINUTES = HIST_DAYS * MINUTES_PER_DAY
    STEP_MINUTES = WALKFORWARD * MINUTES_PER_DAY

    fold = 0
    while True:
        tr0 = fold * STEP_MINUTES
        tr1 = tr0 + TRAIN_WINDOW_MINUTES
        if tr1 >= len(df):
            break

        X_tr, y_tr = X_all.iloc[tr0:tr1], y_all[tr0:tr1]

        rng = np.random.default_rng(42 + fold)
        keep_indices = rng.choice(len(X_tr), size=min(len(X_tr), 500_000), replace=False)
        X_b, y_b = X_tr.iloc[keep_indices], y_tr[keep_indices]
        print(f"Fold {fold}: Training regressor on {len(X_b)} samples...")

        edge = np.abs(df.loc[X_b.index, "next_ret"]) - COST_RT
        sample_weight = np.maximum(edge, 1e-6)

        rf = RandomForestRegressor(**FOREST_PARAMS)
        rf.fit(X_b, y_b, sample_weight=sample_weight)

        mdl_path = os.path.join(MODEL_DIR, f"rf_fold{fold}.joblib")
        joblib.dump(rf, mdl_path)
        final_model = rf
        fold += 1
    
    if final_model:
        best_src  = os.path.join(MODEL_DIR, f"rf_fold{fold-1}.joblib")
        best_link = os.path.join(MODEL_DIR, "rf_best.joblib")
        try:
            if os.path.lexists(best_link):
                os.remove(best_link)
            os.symlink(os.path.basename(best_src), best_link)
            print(f"\n⭐ Last fold {fold-1} linked as rf_best.joblib")
        except Exception as e:
            print(f"Could not create symlink: {e}")

    return final_model