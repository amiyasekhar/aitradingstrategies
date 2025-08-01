#!/usr/bin/env python3
"""
Live AI trading bot using a Random Forest Regressor.
"""
print("🚀 live_trader.py is running 🚀")

import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None

import asyncio, numpy as np, joblib, os
from datetime import datetime
from data_engineering import create_exchange, get_latest_data, add_indicators
from config import PAIR, POSITION_SIZE, WINDOW, TESTNET_MODE, MODEL_DIR, FEE, SLIPPAGE

class LiveTradingBot:
    def __init__(self):
        print("🤖 Initializing AI Trading Bot…")
        
        self.rf_model = joblib.load(f"{MODEL_DIR}/rf_best.joblib")
        print("✅ RF Regressor Model loaded successfully")

        self.profit_threshold = FEE + SLIPPAGE

        self.exchange = create_exchange()
        print(f"🔗 Connected to Binance {'🏖️ TESTNET' if TESTNET_MODE else '🔴 LIVE'}")
        print(f"📊 Pair: {PAIR} | Position Size: {POSITION_SIZE} {PAIR.split('/')[0]}")
        print(f"🎯 Minimum Profit Threshold: {self.profit_threshold:.4%}")
        print("-" * 70)

        self.historical = get_latest_data(WINDOW)
        if self.historical is None or len(self.historical) < WINDOW:
            raise RuntimeError("❌ Unable to fetch initial data window")
        self.historical = add_indicators(self.historical)
        print(f"✅ Loaded & processed initial {WINDOW} bars")

        self.current_position = 0
        self.trades_today = 0
        self.start_balance = None

    async def get_current_balance(self, quote_asset='USDT'):
        try:
            balance = self.exchange.fetch_balance()
            return balance['total'].get(quote_asset, 0.0)
        except Exception as e:
            print(f"❌ Error fetching balance: {e}")
            return None

    async def execute_trade(self, desired_pos: int):
        trade_executed = False
        if self.current_position == 0 and desired_pos == 1:
            print("📈 Signal: FLAT → LONG. Placing BUY order.")
            self.exchange.create_market_buy_order(PAIR, POSITION_SIZE)
            self.current_position = 1
            trade_executed = True
        elif self.current_position == 0 and desired_pos == -1:
            print("📉 Signal: FLAT → SHORT. Placing SELL order.")
            self.exchange.create_market_sell_order(PAIR, POSITION_SIZE)
            self.current_position = -1
            trade_executed = True
        elif self.current_position == 1 and desired_pos != 1:
            print("🚪 Signal: LONG → FLAT. Placing SELL order to close.")
            self.exchange.create_market_sell_order(PAIR, POSITION_SIZE)
            self.current_position = 0
            trade_executed = True
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
            latest_bar = self.exchange.fetch_ohlcv(PAIR, "1m", limit=2)[0]
            new_row = pd.DataFrame([latest_bar], columns=["timestamp", "open", "high", "low", "close", "volume"])
            new_row["timestamp"] = pd.to_datetime(new_row["timestamp"], unit="ms", utc=True)
            new_row = new_row.set_index("timestamp")

            if new_row.index[0] not in self.historical.index:
                self.historical = pd.concat([self.historical.iloc[1:], new_row])
            
            self.historical = add_indicators(self.historical.copy())
            
            X = self.historical[self.rf_model.feature_names_in_].iloc[-1:]
            
            predicted_return = self.rf_model.predict(X)[0]
            
            desired_pos = 0
            if predicted_return > self.profit_threshold:
                desired_pos = 1
            elif predicted_return < -self.profit_threshold:
                desired_pos = -1

            price = self.historical["close"].iloc[-1]
            print(f"🧠 Predicted Return: {predicted_return:+.4%} → Desired Position: {desired_pos} | Price: ${price:,.2f}")
            return desired_pos

        except Exception as e:
            print(f"❌ Error during signal generation: {e}")
            return self.current_position

    async def trading_loop(self):
        print("🚀 Starting live trading loop (Ctrl+C to stop)\n")
        self.start_balance = await self.get_current_balance()
        print(f"💼 Starting Balance: ${self.start_balance:,.2f} USDT")

        while True:
            try:
                desired_pos = await self.get_desired_position()
                await self.execute_trade(desired_pos)
                
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