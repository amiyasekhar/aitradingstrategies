# config.py
"""
Global configuration & hyper-parameters.

Last updated 2025-07-27
────────────────────────────────────────────────────────
• Added TRADE_COST_WEIGHT (required by model_rf.py).
• FOREST_PARAMS block retained.
• RL and fee settings unchanged.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _strip_quotes(val: str | None) -> str | None:
    return val.strip("'\"") if val else None


# ── API credentials ────────────────────────────────────────────────
BINANCE_API_KEY = _strip_quotes(os.getenv("BINANCE_API_KEY"))
BINANCE_SECRET = _strip_quotes(os.getenv("BINANCE_SECRET"))
TESTNET_MODE = os.getenv("TESTNET_MODE", "1") == "1"

# ── Trading parameters ─────────────────────────────────────────────
PAIR = os.getenv("TRADING_PAIR", "BTC/USDT")
TIMEFRAME = "1m"
POSITION_SIZE = float(os.getenv("POSITION_SIZE", "0.001"))
HIST_DAYS = int(os.getenv("HIST_DAYS", "365"))

# ── Indicator & RL parameters ──────────────────────────────────────
WINDOW = int(os.getenv("WINDOW", "60"))
REWARD_SCALE = float(os.getenv("REWARD_SCALE", "1"))
DRAWDOWN_LIMIT = float(os.getenv("DRAWDOWN_LIMIT", "0.10"))
WALKFORWARD = int(os.getenv("WALKFORWARD", "90"))

# ── Discrete-signal thresholds (Random-Forest inference) ───────────
THRESH_UP = float(os.getenv("THRESH_UP", "0.55"))
THRESH_DN = float(os.getenv("THRESH_DN", "0.45"))
PROBA_GAP = float(os.getenv("PROBA_GAP", "0.05"))

# ── Fees & slippage ────────────────────────────────────────────────
TAKER_FEE = float(os.getenv("TAKER_FEE", "0.001"))      # 0.10 %
MAKER_FEE = float(os.getenv("MAKER_FEE", "0.001"))
SLIPPAGE  = float(os.getenv("SLIPPAGE",  "0.0001"))     # 1 bp
FEE = TAKER_FEE

# ── Miscellaneous  (needed by model_rf.py) ─────────────────────────
TRADE_COST_WEIGHT = float(os.getenv("TRADE_COST_WEIGHT", "2.0"))

MODEL_DIR = os.getenv("MODEL_DIR", "models")


def print_config() -> None:
    print("🔧 Configuration loaded:")
    print(f"   🔑 BINANCE_API_KEY: {BINANCE_API_KEY}")
    print(f"   🔒 BINANCE_SECRET:  {BINANCE_SECRET}")
    print(f"   📊 Pair: {PAIR} on {'🏖️ TESTNET' if TESTNET_MODE else '🔴 LIVE'}")
    print(f"   🤖 Position size: {POSITION_SIZE} {PAIR.split('/')[0]}")
    print(
        f"   🎯 Thresholds: Buy>{THRESH_UP:.2f}, "
        f"Sell<{THRESH_DN:.2f}, Gap>±{PROBA_GAP:.2f}"
    )
    print(f"   📁 Model directory: {MODEL_DIR}")


# ── Random-Forest hyper-parameters (for model_rf.py) ───────────────
def _env_override(d: dict[str, object], prefix: str = "RF_") -> dict[str, object]:
    out = d.copy()
    for k, default in d.items():
        v = os.getenv(f"{prefix}{k}")
        if v is not None:
            try:
                out[k] = type(default)(v)     # type: ignore[arg-type]
            except Exception:
                out[k] = v
    return out


FOREST_PARAMS: dict[str, object] = _env_override(
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42,
    }
)