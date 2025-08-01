# data_engineering.py
"""
Public-endpoint-only downloader + feature indicator builder.
"""

from __future__ import annotations
import time, types
from datetime import datetime, timedelta, timezone
import ccxt
from ccxt.base.errors import NetworkError
import pandas as pd
import pandas_ta as ta

from config import PAIR, TIMEFRAME, HIST_DAYS

def _binance() -> ccxt.binance:
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    ex.load_markets()
    return ex

def _utc_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def fetch_history(days: int = HIST_DAYS, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    ex = _binance()
    ex.load_markets()
    
    if start_date:
        since = ex.parse8601(f"{start_date}T00:00:00Z")
    else:
        since = _utc_ms(datetime.utcnow() - timedelta(days=days))

    limit = 1000
    all_ohlcv = []

    while True:
        ohlcv = ex.fetch_ohlcv(PAIR, timeframe=TIMEFRAME, since=since, limit=limit)
        if not ohlcv:
            break
        
        since = ohlcv[-1][0] + 1
        all_ohlcv.extend(ohlcv)

        if end_date and since > ex.parse8601(f"{end_date}T23:59:59Z"):
            break
        
        time.sleep(ex.rateLimit / 1000)

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    if end_date:
        df = df[df['timestamp'] <= pd.to_datetime(end_date, utc=True)]
        
    return add_indicators(df.set_index("timestamp"))

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with technical indicators calculated
    over longer time periods to match the 60-minute prediction horizon.
    """

    def first_col(cols, prefix):
        for c in cols:
            if c.lower().startswith(prefix.lower()):
                return c
        raise KeyError(f"missing indicator starting with '{prefix}'")

    out = df.copy()

    # Calculate indicators with longer lengths
    out["rsi"] = ta.rsi(out["close"], length=60)

    macd = ta.macd(out["close"])
    out["macd"]        = macd[first_col(macd.columns, "MACD_")]
    out["macd_signal"] = macd[first_col(macd.columns, "MACDs")]
    out["macd_hist"]   = macd[first_col(macd.columns, "MACDh")]

    out["atr"]  = ta.atr(out["high"], out["low"], out["close"], length=60)
    out["tema"] = ta.tema(out["close"], length=120)

    bb = ta.bbands(out["close"], length=60, std=2)
    out["bb_upper"]  = bb[first_col(bb.columns, "BBU_")]
    out["bb_middle"] = bb[first_col(bb.columns, "BBM_")]
    out["bb_lower"]  = bb[first_col(bb.columns, "BBL_")]

    out["vwma"] = ta.vwma(out["close"], out["volume"], length=60)
    out["sma"] = ta.sma(out["close"], length=60)
    out["ema"] = ta.ema(out["close"], length=60)
    out["hma"] = ta.hma(out["close"], length=60)

    stoch = ta.stoch(out["high"], out["low"], out["close"], k=60, d=3)
    out["stoch_k"] = stoch[first_col(stoch.columns, "STOCHk")]
    out["stoch_d"] = stoch[first_col(stoch.columns, "STOCHd")]

    out["mom"] = ta.mom(out["close"], length=60)
    adx        = ta.adx(out["high"], out["low"], out["close"], length=60)
    out["adx"] = adx[first_col(adx.columns, "ADX_")]

    out["roc"]  = ta.roc(out["close"], length=60)
    out["cci"]  = ta.cci(out["high"], out["low"], out["close"], length=60)
    out["obv"]  = ta.obv(out["close"], out["volume"])
    out["willr"] = ta.willr(out["high"], out["low"], out["close"], length=60)
    out["mfi"] = ta.mfi(out["high"], out["low"], out["close"], out["volume"], length=60)

    out.dropna(inplace=True)

    col_order = [
        "open","high","low","close","volume","rsi","macd","macd_signal","macd_hist",
        "atr","tema","bb_upper","bb_middle","bb_lower","vwma","sma","ema","hma",
        "stoch_k","stoch_d","mom","adx","roc","cci","obv","willr","mfi",
    ]
    out = out[col_order]
    assert out.shape[1] == 27, f"Expected 27 features, got {out.shape[1]}"
    return out