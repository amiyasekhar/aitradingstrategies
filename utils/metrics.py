# utils/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

def pbo(scores: np.ndarray) -> float:
    """
    Probability of Back-test Over-Fit.
    """
    k = len(scores)
    rank = np.argsort(scores)
    w = np.where(rank == k - 1)[0][0]
    return (w + 1) / (k + 1)

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

def compute_testing_period(equity_series: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Returns the (start, end) timestamps of the backtest.
    """
    start = equity_series.index.min()
    end   = equity_series.index.max()
    return start, end

def compute_total_return(equity_series: pd.Series) -> float:
    """
    Computes total return over the backtest: final/initial − 1.
    """
    return float(equity_series.iloc[-1] / equity_series.iloc[0] - 1)

def compute_trade_stats(actions: np.ndarray,
                        price_series: pd.Series) -> tuple[int, float]:
    """
    Computes number of trades and win rate.
    A trade = any non-zero action.
    A win: for long (1), next-return > 0; for short (2), next-return < 0.
    """
    trade_mask = actions != 0
    n_trades   = int(trade_mask.sum())
    next_ret   = price_series.pct_change().shift(-1).fillna(0)
    wins = ((actions[trade_mask] == 1) & (next_ret[trade_mask] > 0)) | \
           ((actions[trade_mask] == 2) & (next_ret[trade_mask] < 0))
    win_rate   = float(wins.mean()) if n_trades > 0 else 0.0
    return n_trades, win_rate

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

    Returns:
      accuracy,
      macro_precision,
      macro_recall,
      per_class_metrics: dict[class_label] -> (precision, recall)
    """
    # True labels: next-minute direction
    next_ret = price_series.pct_change().shift(-1).fillna(0).values
    true_dir = np.sign(next_ret).astype(int)  # -1, 0, +1

    # Predicted labels from actions
    pred_dir = np.where(actions == 1,  1,
                np.where(actions == 2, -1, 0))

    # 1) Overall accuracy
    accuracy = accuracy_score(true_dir, pred_dir)

    # 2) Macro-average precision & recall across classes [-1, 0, +1]
    labels     = [-1, 0, 1]
    macro_prec = precision_score(true_dir, pred_dir,
                                 labels=labels,
                                 average="macro",
                                 zero_division=0)
    macro_rec  = recall_score(true_dir, pred_dir,
                              labels=labels,
                              average="macro",
                              zero_division=0)

    # 3) Per-class metrics
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