import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
import pandas_ta as ta
from datetime import datetime, timedelta
import os
import requests
from dotenv import load_dotenv

load_dotenv()
CRYPTOQUANT_API_KEY = os.getenv("CRYPTOQUANT_API_KEY")

def run_wfo_for_timeframe(timeframe, params):
    symbol = 'BTC/USDT'
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    fees = 0.001
    initial_capital = 1000.0
    position_size_fraction = 0.2
    train_days = 60
    test_days = 15
    
    max_hold_bars = params['max_hold_bars']
    threshold = params['threshold']
    stop_loss = params['stop_loss']
    take_profit_pct = params['take_profit_pct']
    prob_threshold = params['prob_threshold']
    atr_threshold = params['atr_threshold']

    METRIC_CONFIG = {
        'netflow': {'path': 'exchange-flows/netflow', 'params': {'exchange': 'all_exchange'}, 'data_key': 'netflow_total'},
        'active_addresses': {'path': 'network-data/addresses-count', 'params': {}, 'data_key': 'addresses_count_active'},
        'whale_ratio': {'path': 'flow-indicator/exchange-whale-ratio', 'params': {'exchange': 'all_exchange'}, 'data_key': 'exchange_whale_ratio'}
    }

    exchange = ccxt.binance({'enableRateLimit': True})

    def fetch_cryptoquant_data(metric_name, config, start_date):
        path = config['path']
        print(f"Fetching CryptoQuant data for: {path}...")
        api_url = f"https://api.cryptoquant.com/v1/btc/{path}"
        headers = {'Authorization': f'Bearer {CRYPTOQUANT_API_KEY}'}
        formatted_date = start_date.replace('-', '')
        api_params = {'window': 'day', 'from': formatted_date}
        api_params.update(config['params'])
        response = requests.get(api_url, headers=headers, params=api_params)
        if response.status_code == 200:
            data = response.json().get('result', {}).get('data', [])
            if not data: print(f"Warning: No data returned for {path}"); return pd.DataFrame()
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['date'])
            df = df.set_index('timestamp')
            data_key = config['data_key']
            if data_key in df.columns:
                return df[[data_key]].rename(columns={data_key: metric_name})
            else:
                if len(df.columns) == 1: return df.rename(columns={df.columns[0]: metric_name})
                return pd.DataFrame()
        else:
            print(f"Error fetching data for {path}: {response.status_code} - {response.text}")
            return None

    def fetch_all_ohlcv(exchange, symbol, timeframe, since):
        ohlcv = []; limit = 1000
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
    if not bars: return None
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    for metric_name, config in METRIC_CONFIG.items():
        onchain_df = fetch_cryptoquant_data(metric_name, config, start_date)
        if onchain_df is not None and not onchain_df.empty:
            df = pd.merge(df, onchain_df, left_index=True, right_index=True, how='left')
    
    for col in METRIC_CONFIG.keys():
        if col in df.columns:
            df[col].fillna(method='ffill', inplace=True)

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

    df['future_return'] = df['return'].shift(-1)
    df['label'] = np.where(df['future_return'] > threshold, 1, np.where(df['future_return'] < -threshold, -1, 0))
    df.dropna(inplace=True)

    features = ['atr', 'log_return', 'lagged_return_5', 'rsi14', 'rsi30', 'willr', 'adx'] + list(METRIC_CONFIG.keys())
    features = [f for f in features if f in df.columns]
    
    if len(features) < 8: return None

    def tune_xgboost(X, y):
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1], 'subsample': [0.8, 1.0]}
        model = XGBClassifier(random_state=42, eval_metric='mlogloss')
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X, y)
        print(f"Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    # --- UPDATED: Backtest function now calculates and returns all metrics ---
    def backtest(df, model, period_name):
        if df.empty: return (0,) * 8
        df = df.copy()
        X = df[features]
        probs = model.predict_proba(X)
        df['prob_up'] = probs[:, 2] 
        df['predicted'] = model.predict(X) - 1
        
        position = 0; capital = initial_capital; entry_price = 0.0; trade_capital = 0.0; hold_counter = 0;
        portfolio_values = pd.Series(index=df.index, dtype=float).fillna(capital)
        trades = []

        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            if position == 1:
                pnl_ratio = (current_price / entry_price) - 1
                hold_counter += 1
                if (pnl_ratio < stop_loss or hold_counter >= max_hold_bars or 
                    df['prob_up'].iloc[i] < prob_threshold or pnl_ratio >= take_profit_pct):
                    return_on_trade = pnl_ratio
                    if pnl_ratio < stop_loss: return_on_trade = stop_loss
                    if pnl_ratio >= take_profit_pct: return_on_trade = take_profit_pct
                    profit_loss = trade_capital * return_on_trade
                    capital += profit_loss - (trade_capital + profit_loss) * (fees / 2)
                    position = 0
                    trades.append(profit_loss)
            if position == 0:
                if df['prob_up'].iloc[i] > prob_threshold and df['atr'].iloc[i] < atr_threshold:
                    position = 1; entry_price = current_price; trade_capital = capital * position_size_fraction;
                    capital -= trade_capital * (fees / 2); hold_counter = 0
            if position == 1:
                unrealized_pnl = trade_capital * ((current_price / entry_price) - 1)
                portfolio_values.iloc[i] = (capital - trade_capital) + trade_capital + unrealized_pnl
            else: portfolio_values.iloc[i] = capital
                
        total_return = (portfolio_values.iloc[-1] - initial_capital) / initial_capital
        daily_returns = portfolio_values.pct_change().dropna()
        annualization_factor = 365 * 24 * 60 / (df.index.to_series().diff().mean().total_seconds()/60)
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(annualization_factor) if daily_returns.std() != 0 else 0
        bh_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
        
        successful_trades = sum(1 for t in trades if t > 0)
        failed_trades = len(trades) - successful_trades
        
        y_true = df['label']; y_pred = df['predicted']
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        print(f"\n{period_name} Results:")
        print(f"Total Return: {total_return:.2%} (vs Buy-and-Hold: {bh_return:.2%})")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Successful Trades: {successful_trades} | Failed Trades: {failed_trades}")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        return total_return, sharpe, successful_trades, failed_trades, accuracy, precision, recall

    # WFO Loop
    start_dt = pd.to_datetime(start_date)
    end_dt = df.index.max()
    results = []
    current_start = start_dt
    train_period = timedelta(days=train_days)
    test_period = timedelta(days=test_days)
    while current_start + train_period + test_period <= end_dt:
        train_end = current_start + train_period
        test_end = train_end + test_period
        train_df = df.loc[current_start:train_end]
        test_df = df.loc[train_end:test_end]
        if train_df.empty or test_df.empty or not all(f in train_df.columns for f in features):
            print(f"Skipping window ending {train_end} due to missing data.")
            current_start += test_period; continue
        X_train = train_df[features]; y_train = train_df['label'] + 1
        print(f"\nTuning model for training period ending {train_end}...")
        model = tune_xgboost(X_train, y_train)
        ret, shp, s_trades, f_trades, acc, prec, rec = backtest(test_df, model, f"WFO Window: {train_end} to {test_end}")
        results.append({'return': ret, 'sharpe': shp, 'successful': s_trades, 'failed': f_trades, 
                        'accuracy': acc, 'precision': prec, 'recall': rec})
        current_start += test_period
    if results:
        avg_results = pd.DataFrame(results).mean().to_dict()
        print("\nAverage WFO Results:")
        for key, val in avg_results.items():
            print(f"  Average {key.replace('_', ' ').title()}: {val:.2f}")
        return avg_results
    return None