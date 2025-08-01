import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
import pandas_ta as ta
from datetime import datetime, timedelta

# Parameters
symbol = 'BTC/USDT'
timeframe = '1m'
start_date = '2022-05-01'
threshold = 0.0005
fees = 0.001
initial_capital = 1000.0
position_size_fraction = 0.2
stop_loss = -0.005
max_hold_minutes = 5
prob_threshold = 0.55
train_days = 180
test_days = 30
# --- ADDED: Volatility threshold for regime filter ---
atr_threshold = 33.0

# Initialize exchange
exchange = ccxt.binance({'enableRateLimit': True})

def fetch_all_ohlcv(exchange, symbol, timeframe, since):
    # This function is unchanged
    ohlcv = []
    limit = 1000
    while True:
        try:
            new_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not new_ohlcv: break
            ohlcv.extend(new_ohlcv)
            since = new_ohlcv[-1][0] + 1
        except Exception as e:
            print(f"Error fetching data: {e}"); break
    return ohlcv

since = exchange.parse8601(f'{start_date}T00:00:00Z')
bars = fetch_all_ohlcv(exchange, symbol, timeframe, since)
df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Features (Full set is calculated for potential use in filters)
df['return'] = df['close'].pct_change()
df['log_return'] = np.log(df['close']).diff()
df['lagged_return_1'] = df['return'].shift(1)
df['lagged_return_5'] = df['return'].rolling(5).mean().shift(1)
df['rsi14'] = ta.rsi(df['close'], length=14)
df['rsi30'] = ta.rsi(df['close'], length=30)
df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
df['atr'] = ta.atr(df['high'], df['low'], df['close'])
df['willr'] = ta.willr(df['high'], df['low'], df['close'])
df.dropna(inplace=True)

# Labels
df['future_return'] = df['return'].shift(-1)
df['label'] = np.where(df['future_return'] > threshold, 1,
                       np.where(df['future_return'] < -threshold, -1, 0))
df.dropna(inplace=True)

# --- UPDATED: Simplified feature list based on importance analysis ---
features = ['atr', 'log_return', 'lagged_return_5', 'rsi14', 'rsi30', 'willr', 'adx']

def tune_xgboost(X, y):
    # This function is unchanged
    param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1], 'subsample': [0.8, 1.0]}
    model = XGBClassifier(random_state=42, eval_metric='mlogloss')
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X, y)
    print(f"Best params: {grid_search.best_params_}")
    return grid_search.best_estimator_

def backtest(df, model, period_name):
    if df.empty: return (0,) * 6
    df = df.copy()
    X = df[features]
    probs = model.predict_proba(X)
    df['prob_up'] = probs[:, 2] # Probability of class 1 (up)
    
    position = 0; capital = initial_capital; entry_price = 0.0; trade_capital = 0.0; hold_counter = 0;
    portfolio_values = pd.Series(index=df.index, dtype=float).fillna(capital); trade_count = 0
    
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        
        if position == 1:
            pnl_ratio = (current_price / entry_price) - 1
            hold_counter += 1
            is_stop_loss = pnl_ratio < stop_loss
            is_max_hold = hold_counter >= max_hold_minutes
            
            # Using a simplified model exit for long-only: just check if 'up' prob drops
            is_model_exit = df['prob_up'].iloc[i] < prob_threshold 
            
            if is_stop_loss or is_max_hold or is_model_exit:
                return_on_trade = pnl_ratio if not is_stop_loss else stop_loss
                profit_loss = trade_capital * return_on_trade
                capital += profit_loss - (trade_capital + profit_loss) * (fees / 2)
                position = 0; trade_count += 1
        
        if position == 0:
            # --- UPDATED: Added volatility regime filter to entry logic ---
            passes_vol_filter = df['atr'].iloc[i] < atr_threshold
            
            if df['prob_up'].iloc[i] > prob_threshold and passes_vol_filter:
                position = 1; entry_price = current_price; trade_capital = capital * position_size_fraction;
                capital -= trade_capital * (fees / 2); hold_counter = 0
        
        if position == 1:
            unrealized_pnl = trade_capital * ((current_price / entry_price) - 1)
            portfolio_values.iloc[i] = (capital - trade_capital) + trade_capital + unrealized_pnl
        else: portfolio_values.iloc[i] = capital
            
    total_return = (portfolio_values.iloc[-1] - initial_capital) / initial_capital
    daily_returns = portfolio_values.pct_change().dropna().resample('D').sum()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() != 0 else 0
    bh_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    
    print(f"\n{period_name} Results:")
    print(f"Total Return: {total_return:.2%} (vs Buy-and-Hold: {bh_return:.2%})")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Trade Count: {trade_count}")
    
    return total_return, sharpe, 0, 0, 0, trade_count # ML metrics removed for simplicity

# Main WFO Loop
start_dt = pd.to_datetime(start_date)
end_dt = df.index.max()
results = []
current_start = start_dt

while current_start + timedelta(days=train_days + test_days) <= end_dt:
    train_end = current_start + timedelta(days=train_days)
    test_end = train_end + timedelta(days=test_days)
    train_df = df.loc[current_start:train_end]
    test_df = df.loc[train_end:test_end]
    if train_df.empty or test_df.empty:
        current_start += timedelta(days=test_days); continue
    
    X_train = train_df[features]; y_train = train_df['label'] + 1
    print(f"\nTuning model for training period ending {train_end}...")
    model = tune_xgboost(X_train, y_train)
    ret, shp, acc, prec, rec, trades = backtest(test_df, model, f"WFO Window: {train_end} to {test_end}")
    results.append({'return': ret, 'sharpe': shp, 'trade_count': trades})
    current_start += timedelta(days=test_days)

if results:
    avg_results = pd.DataFrame(results).mean()
    print("\nAverage WFO Results:")
    print(f"  Average Return: {avg_results['return']:.2%}")
    print(f"  Average Sharpe: {avg_results['sharpe']:.2f}")
    print(f"  Average Trades per Period: {avg_results['trade_count']:.0f}")