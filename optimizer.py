import itertools
import pandas as pd
from ingestion.data_ingestion import fetch_historical_data
from strategies.mean_reversion import apply_rsi_strategy
from engine.backtester import SimpleBacktester

def run_optimization(ticker="RELIANCE.NS", period="15y"):
    print("==================================================")
    print(f"🧬 T_RAIDER PARAMETER OPTIMIZER: {ticker}")
    print("==================================================")
    
    print(f"Fetching {period} of data...")
    data = fetch_historical_data([ticker], period=period)
    price_series = data[ticker].dropna()
    
    # 1. Define the "Grid" of parameters we want to test
    rsi_windows = [10, 14, 21]               # How fast the RSI reacts
    buy_thresholds = [25, 30, 35]            # How "oversold" it needs to be to buy
    sell_thresholds = [65, 70, 75]           # How "overbought" it needs to be to sell
    stop_losses = [0.05, 0.08, 0.10, 0.15]   # 5%, 8%, 10%, 15% emergency exits
    
    # Generate every possible combination of the above lists
    combinations = list(itertools.product(rsi_windows, buy_thresholds, sell_thresholds, stop_losses))
    print(f"\nTesting {len(combinations)} different combinations. Please wait...\n")
    
    results = []
    
    # 2. Run the simulation for every single combination
    for window, buy, sell, sl in combinations:
        # Apply the specific strategy parameters
        strategy_df = apply_rsi_strategy(price_series, window=window, buy_threshold=buy, sell_threshold=sell)
        
        # Apply the specific stop loss
        bt = SimpleBacktester(initial_capital=100000.0, stop_loss_pct=sl)
        final_df, _ = bt.run(strategy_df)
        
        # Extract metrics
        metrics = bt.get_metrics(final_df)
        
        # Convert the string percentages back to raw numbers for sorting
        ann_return = float(metrics['Annualized Return'].strip('%'))
        max_dd = float(metrics['Max Drawdown'].strip('%'))
        
        results.append({
            'RSI_Window': window,
            'Buy_At': buy,
            'Sell_At': sell,
            'Stop_Loss': f"{sl*100:.0f}%",
            'Return_%': ann_return,
            'Drawdown_%': max_dd
        })

    # 3. Sort the results to find the highest return
    results_df = pd.DataFrame(results)
    
    # Sort by Highest Return, and if there's a tie, by the Lowest Drawdown
    top_results = results_df.sort_values(by=['Return_%', 'Drawdown_%'], ascending=[False, False]).head(10)
    
    print("🏆 TOP 10 PARAMETER COMBINATIONS 🏆")
    print(top_results.to_string(index=False))
    print("\n==================================================")

if __name__ == "__main__":
    # You can change the ticker here to test other stocks
    run_optimization("RELIANCE.NS")