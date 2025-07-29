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
from ccxt.base.errors import NetworkError
import pandas as pd
import pandas_ta as ta   # pip install pandas_ta

from config import (
    TESTNET_MODE, BINANCE_API_KEY, BINANCE_SECRET,
    PAIR, TIMEFRAME, HIST_DAYS
)

# ── patched Binance instance (no private SAPI) ─────────────────────
def _binance() -> ccxt.binance:
    """
    Return a ccxt.binance exchange that:
    1. Tries main-net public endpoints first.
    2. On NetworkError (DNS / timeout), falls back to spot-testnet
       public endpoints automatically.
    """
    kwargs = {
        "enableRateLimit": True,
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_SECRET,
        "options": {"defaultType": "spot"},
    }
    ex = ccxt.binance(kwargs)

    # Block private SAPI routes as before
    ex.has.update({"fetchCurrencies": False, "margin": False, "leveragedTokens": False})
    ex.sapiGetCapitalConfigGetall = lambda params={}: {}
    ex.sapiGetMarginAllPairs = lambda params={}: []
    ex.sapiGetMarginIsolatedAccount = lambda params={}: {}
    for k in (
        "crossMarginPairsData",
        "isolatedMarginPairsData",
        "crossMarginSymbolMap",
        "isolatedMarginSymbolMap",
    ):
        ex.options.setdefault(k, [])

    # Public-only load_markets()
    def _safe_load(self, reload=False, params={}):
        if not reload and getattr(self, "markets", None):
            return self.markets
        info = self.publicGetExchangeInfo(params)
        mkts = self.parse_markets(info["symbols"])
        self.markets = self.index_by(mkts, "symbol")
        self.symbols = list(self.markets.keys())
        return self.markets

    ex.load_markets = types.MethodType(_safe_load, ex)

    # ── attempt main-net; on failure switch to test-net ─────────────
    try:
        ex.load_markets()
    except NetworkError as err:
        print("⚠️  Main-net unreachable, switching to spot-testnet…")
        ex.set_sandbox_mode(True)  # flips URLs to testnet host
        try:
            ex.load_markets()
        except Exception:
            # propagate original error so stack-trace still useful
            raise err

    return ex


def _utc_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


# ── public api ─────────────────────────────────────────────────────
def fetch_history(days: int = HIST_DAYS, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    ex = _binance()
    ex.load_markets()
    
    # Use date range if provided, otherwise default to 'days'
    if start_date:
        since = ex.parse8601(f"{start_date}T00:00:00Z")
    else:
        since = _utc_ms(datetime.utcnow() - timedelta(days=days))

    limit = 1000
    all_ohlcv = []

    while True:
        ohlcv = ex.fetch_ohlcv(PAIR, timeframe=TIMEFRAME, since=since, limit=limit)
        if len(ohlcv) == 0:
            break
        
        since = ohlcv[-1][0] + 1
        all_ohlcv.extend(ohlcv)

        if end_date and since > ex.parse8601(f"{end_date}T23:59:59Z"):
            break
        
        time.sleep(ex.rateLimit / 1000)

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    # Filter final dataframe to exact end_date if provided
    if end_date:
        df = df[df['timestamp'] <= pd.to_datetime(end_date, utc=True)]
        
    return add_indicators(df.set_index("timestamp"))


# ── indicator builder (produces 24 features) ───────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with exactly 27 columns in this order:

        open, high, low, close, volume,               #  5
        rsi,                                           #  6
        macd, macd_signal, macd_hist,                  #  9
        atr,                                           # 10
        tema,                                          # 11
        bb_upper, bb_middle, bb_lower,                 # 14
        vwma,                                          # 15
        sma, ema, hma,                                 # 18
        stoch_k, stoch_d,                              # 20
        mom, adx,                                      # 22
        roc, cci,                                      # 24
        obv,                                           # 25
        willr,                                         # 26
        mfi                                            # 27  ← NEW
    """

    def first_col(cols, prefix):
        for c in cols:
            if c.lower().startswith(prefix.lower()):
                return c
        raise KeyError(f"missing indicator starting with '{prefix}'")

    out = df.copy()

    # 1 – basic TA
    out["rsi"] = ta.rsi(out["close"], length=14)

    macd = ta.macd(out["close"])
    out["macd"]        = macd[first_col(macd.columns, "MACD_")]
    out["macd_signal"] = macd[first_col(macd.columns, "MACDs")]
    out["macd_hist"]   = macd[first_col(macd.columns, "MACDh")]

    out["atr"]  = ta.atr(out["high"], out["low"], out["close"], length=14)
    out["tema"] = ta.tema(out["close"], length=30)

    bb = ta.bbands(out["close"], length=20, std=2)
    out["bb_upper"]  = bb[first_col(bb.columns, "BBU_")]
    out["bb_middle"] = bb[first_col(bb.columns, "BBM_")]
    out["bb_lower"]  = bb[first_col(bb.columns, "BBL_")]

    out["vwma"] = ta.vwma(out["close"], out["volume"], length=20)

    out["sma"] = ta.sma(out["close"], length=20)
    out["ema"] = ta.ema(out["close"], length=20)
    out["hma"] = ta.hma(out["close"], length=20)

    stoch = ta.stoch(out["high"], out["low"], out["close"])
    out["stoch_k"] = stoch[first_col(stoch.columns, "STOCHk")]
    out["stoch_d"] = stoch[first_col(stoch.columns, "STOCHd")]

    out["mom"] = ta.mom(out["close"], length=10)
    adx        = ta.adx(out["high"], out["low"], out["close"])
    out["adx"] = adx[first_col(adx.columns, "ADX_")]

    out["roc"]  = ta.roc(out["close"], length=10)
    out["cci"]  = ta.cci(out["high"], out["low"], out["close"], length=20)
    out["obv"]  = ta.obv(out["close"], out["volume"])
    out["willr"] = ta.willr(out["high"], out["low"], out["close"], length=14)

    # 27 – Money Flow Index
    out["mfi"] = ta.mfi(out["high"], out["low"], out["close"], out["volume"], length=14)

    # drop warm-up NaNs
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
        "willr",
        "mfi",
    ]
    out = out[col_order]
    assert out.shape[1] == 27, f"Expected 27 features, got {out.shape[1]}"
    return out