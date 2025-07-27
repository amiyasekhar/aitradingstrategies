#!/usr/bin/env python3
"""
Live AI trading bot for Binance testnet with real-time execution.
"""
print("🚀 live_trader.py is running 🚀")

# ─── Monkey-patch importlib.metadata ─────────────────────────────
try:
    import importlib.metadata as _m
except ImportError:
    import importlib_metadata as _m
import importlib
importlib.metadata = _m

# ─── Silence warnings ─────────────────────────────────────────────
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ─── Other imports ────────────────────────────────────────────────
import asyncio, numpy as np, joblib
from datetime import datetime
from stable_baselines3 import DQN

from data_engineering import create_exchange, get_latest_data, add_indicators
from config import PAIR, POSITION_SIZE, THRESH_UP, THRESH_DN, WINDOW, TESTNET_MODE

class LiveTradingBot:
    def __init__(self):
        print("🤖 Initializing AI Trading Bot…")
        self.rf_model  = joblib.load("models/rf_minute.pkl")
        self.dqn_model = DQN.load("models/dqn_minute")
        print("✅ Models loaded successfully")

        self.exchange = create_exchange()
        print(f"🔗 Connected to Binance {'🏖️ TESTNET' if TESTNET_MODE else '🔴 LIVE'}")
        print(f"📊 Pair: {PAIR} | Position: {POSITION_SIZE} BTC")
        print(f"🎯 RF > {THRESH_UP:.2f} BUY, RF < {THRESH_DN:.2f} SELL")
        print("-" * 70)

        df = get_latest_data(WINDOW)
        if df is None or len(df) < WINDOW:
            raise RuntimeError("❌ Unable to fetch initial data window")
        add_indicators(df)
        print(f"✅ Loaded & processed initial {WINDOW} bars")
        self.historical = df

        self.trades_today = 0
        self.start_balance = None

    async def get_account_info(self):
        try:
            bal  = self.exchange.fetch_balance()
            btc  = bal["total"].get("BTC", 0.0)
            usdt = bal["total"].get("USDT", 0.0)
            if self.start_balance is None:
                self.start_balance = {"BTC": btc, "USDT": usdt}
            price = await self.get_current_price()
            return {"BTC": btc, "USDT": usdt, "BTC_USD": btc * price}
        except Exception as e:
            print(f"❌ Error fetching balance: {e}")
            return None

    async def get_current_price(self):
        try:
            t = self.exchange.fetch_ticker(PAIR)
            return float(t["last"])
        except Exception as e:
            print(f"❌ Error fetching price: {e}")
            return 0.0

    async def place_order(self, side: str):
        try:
            if side.lower() == "buy":
                o = self.exchange.create_market_buy_order(PAIR, POSITION_SIZE)
            else:
                o = self.exchange.create_market_sell_order(PAIR, POSITION_SIZE)
            print(f"✅ {side.upper()} executed: ID={o.get('id')}, price={o.get('price')}")
            self.trades_today += 1
            return o
        except Exception as e:
            print(f"❌ {side.upper()} order failed: {e}")
            return None

    async def generate_signals(self):
        try:
            print("🔄 [DEBUG] Fetching latest bar…")
            raw = self.exchange.fetch_ohlcv(PAIR, "1m", limit=1)
            if not raw:
                return None, None, None

            new = pd.DataFrame(raw, columns=["ts","open","high","low","close","vol"])
            new["ts"] = pd.to_datetime(new["ts"], unit="ms", utc=True)
            new.set_index("ts", inplace=True)
            new = new.astype(float)

            print("🔄 [DEBUG] Rolling window…")
            self.historical = pd.concat([self.historical, new]).iloc[-WINDOW:]

            try:
                print("🔄 [DEBUG] Updating indicators…")
                add_indicators(self.historical)
            except Exception as ie:
                print(f"⚠️ Indicator update failed: {ie}")

            feat = list(self.rf_model.feature_names_in_)
            for f in feat:
                if f not in self.historical:
                    self.historical[f] = 0.0
            X = self.historical[feat].values

            print("🔄 [DEBUG] Computing RF probability…")
            rf_proba = self.rf_model.predict_proba(X)[:, 1][-1]

            print("🔄 [DEBUG] Computing DQN action…")
            action, _ = self.dqn_model.predict(X.astype(np.float32), deterministic=True)

            price = float(self.historical["close"].iloc[-1])
            print(f"🔄 [DEBUG] rf_proba={rf_proba:.3f}, action={int(action)}, price={price:.2f}")
            return rf_proba, int(action), price

        except Exception as e:
            print(f"❌ Error generating signals: {e}")
            return None, None, None

    async def print_status(self, rf_proba, action, price, signal):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        acct = await self.get_account_info()
        print(f"\n🕒 {ts} | Price: ${price:.2f}")
        print(f"🎯 RF Prob: {rf_proba:.3f} | DQN Action: {action} | Signal: {signal}")
        if acct:
            pnl = (acct["BTC"] - self.start_balance["BTC"]) * price + \
                  (acct["USDT"] - self.start_balance["USDT"])
            print(f"💼 Balances – BTC: {acct['BTC']:.6f}, USDT: {acct['USDT']:.2f}")
            print(f"📈 Session P&L: ${pnl:.2f}")
        print(f"⚔️ Trades today: {self.trades_today}")
        print("-" * 70)

    async def trading_loop(self):
        print("🚀 Starting live trading loop (Ctrl+C to stop)\n")
        while True:
            try:
                print(f"\n🕒 [DEBUG] Loop at {datetime.now().isoformat()}")
                rf_proba, action, price = await self.generate_signals()

                if rf_proba is not None:
                    print("🔄 [DEBUG] Evaluating trade logic…")
                    if action == 1 and rf_proba > THRESH_UP:
                        signal = "BUY"
                        await self.place_order("buy")
                    elif action == 2 and rf_proba < THRESH_DN:
                        signal = "SELL"
                        await self.place_order("sell")
                    else:
                        signal = "HOLD"
                        print("⏸️ HOLD – no trade")
                    await self.print_status(rf_proba, action, price, signal)
                else:
                    print("⚠️ Skipping—no signals")

                print("⏰ Sleeping 60s…")
                await asyncio.sleep(60)

            except KeyboardInterrupt:
                print("\n👋 Stopped by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                await asyncio.sleep(60)

async def main():
    bot = LiveTradingBot()
    await bot.trading_loop()

if __name__ == "__main__":
    asyncio.run(main())