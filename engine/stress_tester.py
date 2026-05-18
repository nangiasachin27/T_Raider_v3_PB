
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_monte_carlo_analysis(trade_returns, starting_capital=100000, num_simulations=5000):
    """
    Simulates 5,000 randomized versions of your trade history.
    
    trade_returns: list of % returns (e.g., [0.02 for 2%, -0.015 for -1.5%])
    """
    if not trade_returns:
        print("❌ No trades to analyze.")
        return

    all_paths = []
    final_values = []

    print(f"🎲 Running {num_simulations} simulations...")

    for _ in range(num_simulations):
        # RESAMPLING: Pick trades randomly with replacement
        # This simulates different sequences and frequency of wins/losses
        simulated_sequence = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
        
        # Calculate equity curve
        # Initial 1.0 followed by cumulative product of (1 + return)
        path = np.insert(np.cumprod(1 + simulated_sequence), 0, 1.0) * starting_capital
        all_paths.append(path)
        final_values.append(path[-1])

    # --- STATISTICAL METRICS ---
    final_values = np.array(final_values)
    mean_outcome = np.mean(final_values)
    median_outcome = np.median(final_values)
    
    # Value at Risk (VaR): What is the 5% worst-case scenario?
    var_95 = np.percentile(final_values, 5)
    
    # Probability of Profit
    prob_profit = (np.sum(final_values > starting_capital) / num_simulations) * 100

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 6))
    
    # Plot first 100 paths for visual context
    for i in range(min(100, num_simulations)):
        plt.plot(all_paths[i], color='gray', alpha=0.1)

    plt.axhline(starting_capital, color='black', linestyle='--', label="Starting Capital")
    plt.axhline(var_95, color='red', linestyle='--', label=f"95% Confidence Floor (₹{var_95:,.0f})")
    plt.axhline(mean_outcome, color='green', linestyle='-', label=f"Mean Outcome (₹{mean_outcome:,.0f})")
    
    plt.title(f"Monte Carlo: 5,000 Alternative Realities\n(Prob. of Profit: {prob_profit:.1f}%)")
    plt.xlabel("Trade Number")
    plt.ylabel("Portfolio Value (₹)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print("\n" + "="*40)
    print("📊 MONTE CARLO STRESS TEST REPORT")
    print("="*40)
    print(f"Starting Capital    : ₹{starting_capital:,.2f}")
    print(f"Mean Ending Wealth  : ₹{mean_outcome:,.2f}")
    print(f"95% Confidence Floor: ₹{var_95:,.2f} (VaR)")
    print(f"Prob. of Profit     : {prob_profit:.1f}%")
    
    if prob_profit < 60:
        print("⚠️ WARNING: Low probability of profit. Strategy may be 'lucky'.")
    elif var_95 < (starting_capital * 0.8):
        print("⚠️ WARNING: High drawdown risk detected in tail scenarios.")
    else:
        print("✅ STABILITY: Strategy shows robust survival metrics.")
    print("="*40)

if __name__ == "__main__":
    # DUMMY DATA FOR TESTING
    # A mix of small wins, small losses, and a few big winners
    dummy_trades = [0.05, -0.02, 0.01, 0.08, -0.03, -0.01, 0.04, 0.12, -0.05, 0.02]
    run_monte_carlo_analysis(dummy_trades)