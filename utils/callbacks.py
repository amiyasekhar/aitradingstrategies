"""
Risk-management callbacks for live trading.
"""

import time
from datetime import datetime, timezone


class DrawdownKillSwitch:
    def __init__(self, limit=0.10):
        self.high = 1.0
        self.limit = limit

    def update(self, equity):
        self.high = max(self.high, equity)
        dd = (self.high - equity) / self.high
        if dd > self.limit:
            raise SystemExit(f"Drawdown {dd:.2%} breached.")


class RateLimitSentinel:
    def __init__(self, max_tokens=90):
        self.max_tokens = max_tokens

    def check(self, info_obj):
        """
        Hyperliquid's Info object includes internal token bucket.
        """
        tokens = info_obj.token_bucket()
        if tokens < self.max_tokens:
            print(f"[{datetime.now(timezone.utc)}] "
                  f"Low token bucket: {tokens}")
            time.sleep(0.5)
