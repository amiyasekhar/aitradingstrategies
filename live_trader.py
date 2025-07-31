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
from data_engineering import create_exchange, get_latest_data, add_indicators
from config import PAIR, POSITION_SIZE, THRESH_UP, THRESH_DN, PROBA_GAP, WINDOW, TESTNET_MODE, MODEL_DIR
from utils.threshold_search import load_threshold

class LiveTradingBot:
    def __init__(self):
        print("🤖 Initializing AI Trading Bot…")
        
        # --- Load the best RF model and its thresholds ---
        self.rf_model = joblib.load(f"{MODEL_DIR}/rf_best.joblib")
        tag = os.path.splitext(os.path.basename(f"{MODEL_DIR}/rf_best.joblib"))[0]
        tuned = load_threshold(MODEL_DIR, tag) or {}
        self.thresh_up = tuned.get("THRESH_UP", THRESH_UP)
        self.thresh_dn = tuned.get("THRESH_DN", THRESH_DN)
        self.proba_gap = tuned.get("PROBA_GAP", PROBA_GAP)
        print("✅ RF Model loaded successfully")

        self.exchange = create_exchange()
        print(f"🔗 Connected to Binance {'🏖️ TESTNET' if TESTNET_MODE else '🔴 LIVE'}")
        print(f"📊 Pair: {PAIR} | Position Size: {POSITION_SIZE} {PAIR.split('/')[0]}")
        print(f"🎯 Thresholds: Buy>{self.thresh_up:.2f}, Sell<{self.thresh_dn:.2f}, Gap>±{self.proba_gap:.2f}")
        print("-" * 70)

        self.historical = get_latest_data(WINDOW)
        if self.historical is None or len(self.historical) < WINDOW:
            raise RuntimeError("❌ Unable to fetch initial data window")
        self.historical = add_indicators(self.historical)
        print(f"✅ Loaded & processed initial {WINDOW} bars")

        self.current_position = 0 # -1 for short, 0 for flat, 1 for long
        self.trades_today = 0
        self.start_balance = None


    async def get_current_balance(self, quote_asset='USDT'):
        try:
            balance = self.exchange.fetch_balance()
            return balance['total'].get(quote_asset, 0.0)
        except Exception as e:
            print(f"❌ Error fetching balance: {e}")
            return None

    async def get_current_price(self):
        try:
            ticker = self.exchange.fetch_ticker(PAIR)
            return float(ticker["last"])
        except Exception as e:
            print(f"❌ Error fetching price: {e}")
            return 0.0

    async def execute_trade(self, desired_pos: int):
        """Places orders to match the desired position."""
        trade_executed = False
        # Case 1: Go from flat to long
        if self.current_position == 0 and desired_pos == 1:
            print("📈 Signal: FLAT → LONG. Placing BUY order.")
            self.exchange.create_market_buy_order(PAIR, POSITION_SIZE)
            self.current_position = 1
            trade_executed = True
        # Case 2: Go from flat to short
        elif self.current_position == 0 and desired_pos == -1:
            print("📉 Signal: FLAT → SHORT. Placing SELL order.")
            self.exchange.create_market_sell_order(PAIR, POSITION_SIZE)
            self.current_position = -1
            trade_executed = True
        # Case 3: Go from long to flat (exit)
        elif self.current_position == 1 and desired_pos != 1:
            print("🚪 Signal: LONG → FLAT. Placing SELL order to close.")
            self.exchange.create_market_sell_order(PAIR, POSITION_SIZE)
            self.current_position = 0
            trade_executed = True
        # Case 4: Go from short to flat (exit)
        elif self.current_position == -1 and desired_pos != -1:
            print("🚪 Signal: SHORT → FLAT. Placing BUY order to close.")
            self.exchange.create_market_buy_order(PAIR, POSITION_SIZE)
            self.current_position = 0
            trade_executed = True
        
        if trade_executed:
            self.trades_today +=1
            print(f"✅ Trade executed. New position: {self.current_position}")
        else:
            print(f"▶️ Holding position: {self.current_position}. No trade needed.")

    async def get_desired_position(self) -> int:
        """Fetches new data and computes the model's desired position."""
        try:
            # Fetch the latest bar and add to historical data
            latest_bar = self.exchange.fetch_ohlcv(PAIR, "1m", limit=2)[0] # fetch 2, take first to ensure it's closed
            new_row = pd.DataFrame([latest_bar], columns=["timestamp", "open", "high", "low", "close", "volume"])
            new_row["timestamp"] = pd.to_datetime(new_row["timestamp"], unit="ms", utc=True)
            new_row = new_row.set_index("timestamp")

            # Check for duplicate index before appending
            if new_row.index[0] not in self.historical.index:
                self.historical = pd.concat([self.historical.iloc[1:], new_row])
            
            # Recalculate indicators
            self.historical = add_indicators(self.historical.copy())
            
            # Get latest features and predict
            X = self.historical[self.rf_model.feature_names_in_].iloc[-1:]
            prob = self.rf_model.predict_proba(X)[0, 1]
            
            # Determine desired position
            gap_m = abs(prob - 0.5) >= self.proba_gap
            desired_pos = 0
            if prob > self.thresh_up and gap_m:
                desired_pos = 1
            elif prob < self.thresh_dn and gap_m:
                desired_pos = -1

            price = self.historical["close"].iloc[-1]
            print(f"🧠 Model probability: {prob:.4f} → Desired Position: {desired_pos} | Price: ${price:,.2f}")
            return desired_pos

        except Exception as e:
            print(f"❌ Error during signal generation: {e}")
            return self.current_position # Return current position on error to avoid false exits

    async def trading_loop(self):
        print("🚀 Starting live trading loop (Ctrl+C to stop)\n")
        
        # Set start balance for P&L tracking
        self.start_balance = await self.get_current_balance()
        print(f"💼 Starting Balance: ${self.start_balance:,.2f} USDT")

        while True:
            try:
                desired_pos = await self.get_desired_position()
                await self.execute_trade(desired_pos)
                
                # Print status
                current_bal = await self.get_current_balance()
                pnl = current_bal - self.start_balance if current_bal and self.start_balance else 0
                print(f"⚔️ Trades today: {self.trades_today} | Session P&L: ${pnl:,.2f}")
                print("-" * 70)

                await asyncio.sleep(60)

            except KeyboardInterrupt:
                print("\n👋 Manual stop detected. Exiting.")
                break
            except Exception as e:
                print(f"❌ An unexpected error occurred in the trading loop: {e}")
                await asyncio.sleep(60)

async def main():
    try:
        bot = LiveTradingBot()
        await bot.trading_loop()
    except Exception as e:
        print(f"FATAL: Bot failed to initialize: {e}")

if __name__ == "__main__":
    asyncio.run(main())