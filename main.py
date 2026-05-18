import json
import os
# Fix: Import the specific function from your existing file
from engine.portfolio_manager import suggest_allocation 
from daily_screener import run_screener
from utils import get_config_tickers

def run_t_raider_terminal():
    # 1. Setup Defaults (Since we aren't tracking a live portfolio file yet)
    STARTING_CASH = 100000.00
    tickers = get_config_tickers()
    
    print("\n" + "="*50)
    print("🏦 T_RAIDER LIVE TERMINAL")
    print("="*50)
    print(f"ESTIMATED CASH AVAILABLE : ₹{STARTING_CASH:,.2f}")
    print("="*50)

    # 2. Run the Hybrid Screener
    # This fetches signals from your 'best' strategies
    buy_signals, sell_signals = run_screener(tickers)

    # 3. THE CLEAN ACTION REPORT
    print("\n" + "🚀 TARGETED ACTIONS FOR TODAY")
    print("-" * 50)

    if not buy_signals:
        print("⏳ STATUS: No high-probability entry points found.")
    else:
        print(f"✅ FOUND {len(buy_signals)} OPPORTUNITIES")
        # 4. Use your existing Risk Allocation logic!
        suggest_allocation(buy_signals, current_cash=STARTING_CASH)

    # 5. MARKET WARNINGS (Ghost Sells)
    # Since we don't track holdings yet, we label these as "Market Warnings"
    if sell_signals:
        print("\n🛑 MARKET WARNINGS (Watch for Trend Reversals):")
        for ticker, price in sell_signals:
            print(f"   {ticker} is flashing SELL at ₹{price:.2f}")
    
    print("-" * 50)
    print("\nTrading Note: Tomorrow (May 1) is a market holiday.")

if __name__ == "__main__":
    run_t_raider_terminal()