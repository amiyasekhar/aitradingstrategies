"""
Walk-forward Random-Forest baseline with class-balancing
and trade-cost weighting.
"""

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
    df = df.copy()
    df["next_ret"] = df["close"].pct_change().shift(-1)
    df["y"] = (df["next_ret"] > 0).astype(int)
    return df.dropna().reset_index(drop=True)


def train_walkforward(df: pd.DataFrame) -> RandomForestClassifier:
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_all = df.select_dtypes(include=[np.number]).drop(columns=["next_ret", "y"])
    y_all = df["y"].values
    times = df.index

    final_model = None
    results = []

    fold = 0
    while True:
        train_start = fold * WALKFORWARD
        train_end = train_start + HIST_DAYS
        test_end = train_end + WALKFORWARD
        if test_end > len(df):
            break

        X_train = X_all.iloc[train_start:train_end]
        y_train = y_all[train_start:train_end]
        X_test = X_all.iloc[train_end:test_end]
        y_test = y_all[train_end:test_end]

        # ── balance classes ───────────────────────────────────────
        rng = np.random.default_rng(42)
        idx_0 = np.where(y_train == 0)[0]
        idx_1 = np.where(y_train == 1)[0]
        min_n = min(len(idx_0), len(idx_1))
        keep_idx = np.concatenate(
            [rng.choice(idx_0, min_n, replace=False),
             rng.choice(idx_1, min_n, replace=False)]
        )
        X_bal = X_train.iloc[keep_idx]
        y_bal = y_train[keep_idx]

        print(f"Fold {fold}: balanced counts 0→{(y_bal==0).sum()}, 1→{(y_bal==1).sum()}")

        model = RandomForestClassifier(**FOREST_PARAMS)
        sample_weight = np.where(y_bal == 1, TRADE_COST_WEIGHT, 1.0)
        model.fit(X_bal, y_bal, sample_weight=sample_weight)

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        results.append(
            {
                "fold": fold,
                "train_start": times[train_start],
                "train_end": times[train_end - 1],
                "test_end": times[test_end - 1],
                "auc": auc,
            }
        )

        joblib.dump(model, os.path.join(MODEL_DIR, f"rf_fold{fold}.joblib"))
        final_model = model
        fold += 1

    print("\nWalk-forward AUC per fold:")
    print(pd.DataFrame(results).to_string(index=False))
    return final_model