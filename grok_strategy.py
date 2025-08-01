import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import pandas_ta as ta
from datetime import datetime

# Parameters
symbol = 'BTC/USDT'
timeframe = '1m'
start_date = '2022-05-01'
in_sample_end = '2022-10-31'
out_sample_start = '2025-01-01'
threshold = 0.0005
fees = 0.001
initial_capital = 1000.0
position_size_fraction = 0.2
stop_loss = -0.005
max_hold_minutes = 5
prob_threshold = 0.6  # For proba-based signals

# Initialize exchange
exchange = ccxt.binance({'enableRateLimit': True})

# Fetch function (unchanged)
def fetch_all_ohlcv(exchange, symbol, timeframe, since):
    ohlcv = []
    limit = 1000
    while True:
        try:
            new_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not new_ohlcv:
                break
            ohlcv.extend(new_ohlcv)
            since = new_ohlcv[-1][0] + 1
            print(f"Fetched {len(ohlcv)} candles up to {datetime.fromtimestamp(since/1000)}")
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    return ohlcv

# Fetch data
since = exchange.parse8601(f'{start_date}T00:00:00Z')
bars = fetch_all_ohlcv(exchange, symbol, timeframe, since)
df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Enhanced features from research
df['return'] = df['close'].pct_change()
df['lagged_return_1'] = df['return'].shift(1)
df['lagged_return_5'] = df['return'].rolling(5).mean().shift(1)
df['rsi14'] = ta.rsi(df['close'], length=14)
df['rsi30'] = ta.rsi(df['close'], length=30)
df['rsi200'] = ta.rsi(df['close'], length=200)
df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
df['ema_5'] = ta.ema(df['close'], length=5)
df['ema_20'] = ta.ema(df['close'], length=20)
df['vol_change'] = df['volume'].pct_change()
bb = ta.bbands(df['close'], length=20)
df['bb_upper'] = bb['BBU_20_2.0']
df['bb_lower'] = bb['BBL_20_2.0']
df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
df['mom30'] = ta.mom(df['close'], length=30)
stoch30 = ta.stoch(df['high'], df['low'], df['close'], k=30, d=3)
df['k30'] = stoch30['STOCHk_30_3_3']
df['d30'] = stoch30['STOCHd_30_3_3']
stoch200 = ta.stoch(df['high'], df['low'], df['close'], k=200, d=3)
df['k200'] = stoch200['STOCHk_200_3_3']
df['d200'] = stoch200['STOCHd_200_3_3']
df['cci'] = ta.cci(df['high'], df['low'], df['close'])
df['atr'] = ta.atr(df['high'], df['low'], df['close'])
df['willr'] = ta.willr(df['high'], df['low'], df['close'])
df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])
df['obv'] = ta.obv(df['close'], df['volume'])
df.dropna(inplace=True)

# Labels
df['future_return'] = df['return'].shift(-1)
df['label'] = np.where(df['future_return'] > threshold, 1,
                       np.where(df['future_return'] < -threshold, -1, 0))
df.dropna(inplace=True)

# Split data
in_sample = df.loc[:in_sample_end]
out_sample = df.loc[out_sample_start:]

# Features list
features = ['lagged_return_1', 'lagged_return_5', 'rsi14', 'rsi30', 'rsi200', 'macd', 'ema_5', 'ema_20', 'vol_change', 
            'bb_upper', 'bb_lower', 'adx', 'mom30', 'k30', 'd30', 'k200', 'd200', 'cci', 'atr', 'willr', 'cmf', 'obv']

# Train XGBoost (map labels to 0,1,2)
X_train = in_sample[features]
y_train = in_sample['label'] + 1  # -1->0, 0->1, 1->2
model = XGBClassifier(n_estimators=50, max_depth=5, scale_pos_weight=3, random_state=42, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Backtest function with proba
def backtest(df, model, period_name):
    if df.empty:
        print(f"\n{period_name} data is empty.")
        return 0, 0, 0, 0, 0
    
    df = df.copy()
    X = df[features]
    probs = model.predict_proba(X)
    df['prob_down'] = probs[:, 0]  # Prob of -1 (mapped 0)
    df['prob_neutral'] = probs[:, 1]  # Prob of 0 (mapped 1)
    df['prob_up'] = probs[:, 2]  # Prob of 1 (mapped 2)
    df['predicted'] = model.predict(X) - 1  # For metrics
    
    position = 0
    capital = initial_capital
    entry_price = 0.0
    trade_capital = 0.0
    hold_counter = 0
    portfolio_values = pd.Series(index=df.index, dtype=float).fillna(capital)
    
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        
        if position == 1:
            pnl_ratio = (current_price / entry_price) - 1
            hold_counter += 1
            is_stop_loss = pnl_ratio < stop_loss
            is_max_hold = hold_counter >= max_hold_minutes
            is_model_exit = df['prob_down'].iloc[i] > prob_threshold
            
            if is_stop_loss or is_max_hold or is_model_exit:
                return_on_trade = pnl_ratio if not is_stop_loss else stop_loss
                profit_loss = trade_capital * return_on_trade
                capital += profit_loss - (trade_capital + profit_loss) * (fees / 2)
                position = 0
                entry_price = 0
                trade_capital = 0
                hold_counter = 0
        
        if position == 0:
            if df['prob_up'].iloc[i] > prob_threshold and df['rsi14'].iloc[i] < 70:
                position = 1
                entry_price = current_price
                trade_capital = capital * position_size_fraction
                capital -= trade_capital * (fees / 2)
                hold_counter = 0
        
        if position == 1:
            unrealized_pnl = trade_capital * ((current_price / entry_price) - 1)
            portfolio_values.iloc[i] = (capital - trade_capital) + trade_capital + unrealized_pnl
        else:
            portfolio_values.iloc[i] = capital
    
    total_return = (portfolio_values.iloc[-1] - initial_capital) / initial_capital
    daily_returns = portfolio_values.pct_change().dropna().resample('D').sum()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() != 0 else 0
    bh_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    
    y_true = df['label']
    y_pred = df['predicted']
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\n{period_name} Results:")
    print(f"Total Return: {total_return:.2%} (vs Buy-and-Hold: {bh_return:.2%})")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return total_return, sharpe, accuracy, precision, recall

# Run
backtest(in_sample, model, "In-Sample")
backtest(out_sample, model, "Out-of-Sample")