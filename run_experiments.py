# run_experiments.py
from grok_strategy import run_wfo_for_timeframe
import pandas as pd

EXPERIMENT_CONFIGS = {
    '5m': {
        'max_hold_bars': 5, 'threshold': 0.0005, 'stop_loss': -0.005,
        'take_profit_pct': 0.002, 'prob_threshold': 0.55, 'atr_threshold': 10.0
    },
    '15m': {
        'max_hold_bars': 5, 'threshold': 0.0008, 'stop_loss': -0.008,
        'take_profit_pct': 0.004, 'prob_threshold': 0.55, 'atr_threshold': 30.0
    },
    '1h': {
        'max_hold_bars': 5, 'threshold': 0.001, 'stop_loss': -0.01,
        'take_profit_pct': 0.005, 'prob_threshold': 0.55, 'atr_threshold': 200.0
    },
    '4h': {
        'max_hold_bars': 5, 'threshold': 0.002, 'stop_loss': -0.02,
        'take_profit_pct': 0.01, 'prob_threshold': 0.55, 'atr_threshold': 800.0
    },
    '1d': {
        'max_hold_bars': 5, 'threshold': 0.005, 'stop_loss': -0.05,
        'take_profit_pct': 0.025, 'prob_threshold': 0.55, 'atr_threshold': 5000.0
    },
}

if __name__ == "__main__":
    all_results = []
    for timeframe, params in EXPERIMENT_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"RUNNING EXPERIMENT FOR TIMEFRAME: {timeframe}")
        print(f"PARAMETERS: {params}")
        print(f"{'='*50}\n")
        
        avg_results = run_wfo_for_timeframe(timeframe, params)
        if avg_results is not None:
            avg_results['timeframe'] = timeframe
            all_results.append(avg_results)

    if all_results:
        summary_df = pd.DataFrame(all_results).set_index('timeframe')
        print("\n\n--- FINAL EXPERIMENT SUMMARY ---")
        
        # --- UPDATED: Format and display all new columns ---
        formatters = {
            'return': '{:.2%}'.format,
            'sharpe': '{:.2f}'.format,
            'successful': '{:.2f}'.format,
            'failed': '{:.2f}'.format,
            'accuracy': '{:.2%}'.format,
            'precision': '{:.2%}'.format,
            'recall': '{:.2%}'.format,
        }
        print(summary_df.to_string(formatters=formatters))