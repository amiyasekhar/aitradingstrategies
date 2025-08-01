# utils/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

def bootstrap_equity(equity_curve: pd.Series,
                     n_iter: int = 10_000) -> tuple[float, float, float]:
    """
    Bootstraps minute-level returns to build a distribution of
    cumulative returns; returns (mu, sigma, VaR95).
    """
    rets = equity_curve.pct_change().dropna().values
    paths = np.random.choice(rets, (n_iter, len(rets)))
    cumu = (1 + paths).prod(axis=1)
    return float(np.mean(cumu)), float(np.std(cumu)), float(np.quantile(cumu, 0.05))

def compute_sharpe_ratio(equity_series: pd.Series,
                         minutes_per_year: float = 252 * 6.5 * 60) -> float:
    """
    Annualized Sharpe ratio based on minute-level returns.
    """
    rets = equity_series.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    return float((rets.mean() / rets.std()) * np.sqrt(minutes_per_year))

def compute_signal_classification_metrics(actions: np.ndarray,
                                          price_series: pd.Series) -> tuple[float, float, float, dict]:
    """
    Three-class evaluation of your signals:
      - Class  1: long (buy)
      - Class -1: short (sell)
      - Class  0: hold
    """
    next_ret = price_series.pct_change().shift(-1).fillna(0).values
    true_dir = np.sign(next_ret).astype(int)

    if -1 in actions:
        pred_dir = actions.astype(int)
    else: 
        pred_dir = np.where(actions == 1, 1, np.where(actions == 2, -1, 0))

    accuracy = accuracy_score(true_dir, pred_dir)

    labels     = [-1, 0, 1]
    macro_prec = precision_score(true_dir, pred_dir,
                                 labels=labels,
                                 average="macro",
                                 zero_division=0)
    macro_rec  = recall_score(true_dir, pred_dir,
                              labels=labels,
                              average="macro",
                              zero_division=0)

    prec_by_class = precision_score(true_dir, pred_dir,
                                    labels=labels,
                                    average=None,
                                    zero_division=0)
    rec_by_class  = recall_score(true_dir, pred_dir,
                                 labels=labels,
                                 average=None,
                                 zero_division=0)
    per_class = {
        lbl: (float(p), float(r))
        for lbl, p, r in zip(labels, prec_by_class, rec_by_class)
    }

    return accuracy, macro_prec, macro_rec, per_class