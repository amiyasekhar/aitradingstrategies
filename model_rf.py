# model_rf.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from config import (
    HIST_DAYS,
    FOREST_PARAMS,
    WALKFORWARD,
    MODEL_DIR,
    TRADE_COST_WEIGHT,
)

def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with a 'close' column,
    - compute next-period return 'next_ret'
    - binary-label 'y' = 1 if next_ret > 0 else 0
    - drop NaNs
    """
    df = df.copy()
    df["next_ret"] = df["close"].pct_change().shift(-1)
    df["y"]        = (df["next_ret"] > 0).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df

def train_walkforward(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Walk-forward train:
      - Train on HIST_DAYS, test on next WALKFORWARD days
      - Penalize 'Up' samples via sample_weight=TRADE_COST_WEIGHT
      - Save each fold's model under MODEL_DIR/rf_fold{fold}.joblib
    Returns the final fold's RandomForestClassifier.
    """
    # Ensure MODEL_DIR exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Prepare features and target
    X_all = df.select_dtypes(include=[np.number]).drop(columns=["next_ret", "y"])
    y_all = df["y"].values
    times = df.index

    final_model = None
    results = []

    fold = 0
    while True:
        train_start = fold * WALKFORWARD
        train_end   = train_start + HIST_DAYS
        test_end    = train_end + WALKFORWARD

        if test_end > len(df):
            break

        X_train = X_all.iloc[train_start:train_end]
        y_train = y_all[train_start:train_end]
        X_test  = X_all.iloc[train_end:test_end]
        y_test  = y_all[train_end:test_end]

        # Initialize and fit with trade-cost weighting
        model = RandomForestClassifier(**FOREST_PARAMS)
        sample_weight = np.where(y_train == 1, TRADE_COST_WEIGHT, 1.0)
        model.fit(X_train, y_train, sample_weight=sample_weight)

        # Evaluate AUC on the test slice
        y_prob = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        results.append({
            "fold":        fold,
            "train_start": times[train_start],
            "train_end":   times[train_end - 1],
            "test_end":    times[test_end - 1],
            "auc":         auc,
        })

        # Save this fold's model
        filename = os.path.join(MODEL_DIR, f"rf_fold{fold}.joblib")
        joblib.dump(model, filename)

        final_model = model
        fold += 1

    # Print a summary of fold AUCs
    summary = pd.DataFrame(results)
    print("\nWalk-forward AUC per fold:")
    print(summary.to_string(index=False))

    return final_model