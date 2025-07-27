# data_engineering.py
"""
Public-endpoint-only downloader + 24-feature indicator builder
— safe for Binance spot-testnet — and robust to pandas-ta version
differences.

Last update 2025-07-27
"""

from __future__ import annotations
import time, types
from datetime import datetime, timedelta, timezone
from typing import List

import ccxt
import pandas as pd
import pandas_ta as ta   # pip install pandas_ta

from config import (
    TESTNET_MODE, BINANCE_API_KEY, BINANCE_SECRET,
    PAIR, TIMEFRAME, HIST_DAYS
)

# ── patched Binance instance (no private SAPI) ─────────────────────
def _binance() -> ccxt.binance:
    ex = ccxt.binance(
        {
            "enableRateLimit": True,
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_SECRET,
            "options": {"defaultType": "spot"},
        }
    )
    if TESTNET_MODE:
        ex.set_sandbox_mode(True)

    # block private routes
    ex.has.update({"fetchCurrencies": False, "margin": False, "leveragedTokens": False})
    ex.sapiGetCapitalConfigGetall = lambda params={}: {}
    ex.sapiGetMarginAllPairs = lambda params={}: []
    ex.sapiGetMarginIsolatedAccount = lambda params={}: {}

    for k in (
        "crossMarginPairsData", "isolatedMarginPairsData",
        "crossMarginSymbolMap", "isolatedMarginSymbolMap",
    ):
        ex.options.setdefault(k, [])

    def _safe_load(self, reload=False, params={}):
        if not reload and getattr(self, "markets", None):
            return self.markets
        info = self.publicGetExchangeInfo(params)
        mkts = self.parse_markets(info["symbols"])
        self.markets = self.index_by(mkts, "symbol"); self.symbols = list(self.markets)
        return self.markets

    ex.load_markets = types.MethodType(_safe_load, ex)
    return ex


def _utc_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


# ── public api ─────────────────────────────────────────────────────
def fetch_history(days: int = HIST_DAYS) -> pd.DataFrame:
    ex = _binance(); ex.load_markets()
    since = _utc_ms(datetime.utcnow() - timedelta(days=days))
    rows: List[List] = []

    while since < _utc_ms(datetime.utcnow()):
        batch = ex.fetch_ohlcv(PAIR, timeframe=TIMEFRAME, since=since, limit=1000)
        if not batch: break
        rows.extend(batch)
        since = batch[-1][0] + 60_000
        time.sleep(ex.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return add_indicators(df.set_index("timestamp"))


# ── indicator builder (produces 24 features) ───────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with exactly 25 columns in this order:

        open, high, low, close, volume,
        rsi,
        macd, macd_signal, macd_hist,
        atr,
        tema,
        bb_upper, bb_middle, bb_lower,
        vwma,
        sma, ema, hma,
        stoch_k, stoch_d,
        mom, adx,
        roc, cci,
        obv            ← NEW (25th feature)
    """

    def first_col(cols, prefix: str):
        for c in cols:
            if c.lower().startswith(prefix.lower()):
                return c
        raise KeyError(f"missing indicator column starting with '{prefix}'")

    out = df.copy()

    # 1  RSI
    out["rsi"] = ta.rsi(out["close"], length=14)

    # 2-4  MACD
    macd = ta.macd(out["close"])
    out["macd"]        = macd[first_col(macd.columns, "MACD_")]
    out["macd_signal"] = macd[first_col(macd.columns, "MACDs")]
    out["macd_hist"]   = macd[first_col(macd.columns, "MACDh")]

    # 5  ATR
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    # 6  TEMA
    out["tema"] = ta.tema(out["close"], length=30)

    # 7-9  Bollinger Bands
    bb = ta.bbands(out["close"], length=20, std=2)
    out["bb_upper"]  = bb[first_col(bb.columns, "BBU_")]
    out["bb_middle"] = bb[first_col(bb.columns, "BBM_")]
    out["bb_lower"]  = bb[first_col(bb.columns, "BBL_")]

    # 10  VWMA
    out["vwma"] = ta.vwma(out["close"], out["volume"], length=20)

    # 11-13  Moving averages
    out["sma"] = ta.sma(out["close"], length=20)
    out["ema"] = ta.ema(out["close"], length=20)
    out["hma"] = ta.hma(out["close"], length=20)

    # 14-15  Stochastic
    stoch = ta.stoch(out["high"], out["low"], out["close"])
    out["stoch_k"] = stoch[first_col(stoch.columns, "STOCHk")]
    out["stoch_d"] = stoch[first_col(stoch.columns, "STOCHd")]

    # 16  Momentum
    out["mom"] = ta.mom(out["close"], length=10)

    # 17  ADX
    adx = ta.adx(out["high"], out["low"], out["close"])
    out["adx"] = adx[first_col(adx.columns, "ADX_")]

    # 18-19  ROC & CCI
    out["roc"] = ta.roc(out["close"], length=10)
    out["cci"] = ta.cci(out["high"], out["low"], out["close"], length=20)

    # 20  OBV  ← missing feature added
    out["obv"] = ta.obv(out["close"], out["volume"])

    # Clean-up
    out.dropna(inplace=True)

    col_order = [
        "open","high","low","close","volume",
        "rsi",
        "macd","macd_signal","macd_hist",
        "atr",
        "tema",
        "bb_upper","bb_middle","bb_lower",
        "vwma",
        "sma","ema","hma",
        "stoch_k","stoch_d",
        "mom","adx",
        "roc","cci",
        "obv",
    ]
    out = out[col_order]
    assert out.shape[1] == 25, f"Expected 25 features, got {out.shape[1]}"
    return out