import pandas as pd
import json
import os
from ingestion.data_ingestion import fetch_historical_data, get_stock_data
from strategies.trend_follower import apply_golden_cross_strategy
from strategies.mean_reversion import apply_rsi_strategy
from strategies.volatility import apply_bollinger_strategy
from strategies.breakout import apply_breakout_strategy
from strategies.momentum import apply_macd_strategy # Added MACD
from engine.backtester import SimpleBacktester

def run_hybrid_global_test():
    # 1. Load Data
    if not os.path.exists('config/stocks.json'):
        print("❌ ERROR: config/stocks.json not found.")
        return

    with open('config/stocks.json', 'r') as f:
        tickers = json.load(f)['nifty_50']
    
    if not os.path.exists('config/optimal_params.json'):
        print("❌ ERROR: optimal_params.json not found. Run auto_optimizer.py first.")
        return

    with open('config/optimal_params.json', 'r') as f:
        optimized_params = json.load(f)

    # 2. Fetch Data
    full_data = fetch_historical_data(tickers, period="5y")
    
    results = []

    print("\n🚀 RUNNING HYBRID BACKTEST (STOCK-BY-STOCK BEST)...")
    print(f"{'TICKER':12} | {'BEST STRAT':15} | {'RETURN':10} | {'STATUS'}")
    print("-" * 55)

    for ticker in tickers:
        plan = optimized_params.get(ticker)
        if not plan:
            print(f"{ticker:12} | {'MISSING':15} | {'N/A':10} | ⚠️ Not optimized")
            continue
            
        df = get_stock_data(full_data, ticker)
        if df.empty or len(df) < 250: 
            continue
            
        # --- FIXED PRICE EXTRACTION ---
        if 'Adj Close' in df.columns:
            price = df['Adj Close']
        elif 'Close' in df.columns:
            price = df['Close']
        else:
            print(f"{ticker:12} | {'ERROR':15} | {'N/A':10} | ❌ No price data")
            continue

        strat_type = plan['strategy']
        params = plan['params']
        
        # 3. Execute the Winning Strategy
        if strat_type == "TREND":
            strat_df = apply_golden_cross_strategy(price)
        elif strat_type == "RSI":
            strat_df = apply_rsi_strategy(price, params['window'], params['buy'], params['sell'])
        elif strat_type == "VOLATILITY":
            strat_df = apply_bollinger_strategy(price)
        elif strat_type == "MACD":
            strat_df = apply_macd_strategy(price, params['fast'], params['slow'], params['signal'])
        elif strat_type == "BREAKOUT":
            # Fixed: Donchian breakout only needs price, not volume
            strat_df = apply_breakout_strategy(price, window=params['window'])
        else: 
            continue

        # 4. Backtest the result
        bt = SimpleBacktester(stop_loss_pct=0.10)
        final_df, _ = bt.run(strat_df)
        metrics = bt.get_metrics(final_df)
        
        try:
            ret_val = float(metrics['Post-Tax Annualized'].replace('%',''))
            results.append(ret_val)
            print(f"{ticker:12} | {strat_type:15} | {ret_val:>8.2f}% | ✅ Done")
        except ValueError:
            print(f"{ticker:12} | {strat_type:15} | {'ERROR':10} | ❌ Metric Fail")

    # 5. Final Summary
    if results:
        avg_ret = sum(results) / len(results)
        print("\n" + "="*50)
        print("📈 FINAL HYBRID PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Average Annual Return (All Stocks): {avg_ret:.2f}%")
        print(f"Total Universe Tracked: {len(results)} stocks")
        print(f"Target Performance (Goal): 15.00%")
        print("="*50)

if __name__ == "__main__":
    run_hybrid_global_test()