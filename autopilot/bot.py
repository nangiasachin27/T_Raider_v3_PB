import os
import sys
import argparse
import pandas as pd
import numpy as np
import json                          # ADD
from pathlib import Path 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from daily_screener import run_screener
from autopilot.logger import (
    load_portfolio, record_transaction,
    _normalise_holding, get_holding_entry_price
)
from utils import get_config_tickers
from ingestion.data_ingestion import fetch_historical_data, get_stock_data

def get_active_capital():
    override = Path("config/capital_override.json")
    if override.exists():
        with open(override) as f:
            return json.load(f).get("total_baseline_wealth", 100000.0)
    return 100000.0

def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_pc = np.abs(df['High'] - df['Close'].shift(1))
    low_pc = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    return tr.rolling(window=window).mean().iloc[-1]


def check_partial_exits(tickers, full_market_data, partial_exit_threshold=0.20,
                        partial_exit_fraction=0.50):
    print(f"\n--- PHASE 0: PARTIAL EXITS (Profit Lock-In at +{partial_exit_threshold*100:.0f}%) ---")

    portfolio = load_portfolio()
    holdings = portfolio.get('holdings', {})

    if not holdings:
        print(" No open positions to check.")
        return 0

    exits_executed = 0

    for ticker, holding_data in list(holdings.items()):
        holding = _normalise_holding(holding_data)
        qty = holding['qty']
        entry_price = holding['entry_price']

        if entry_price <= 0:
            print(f" ⚠️ {ticker} — entry price unknown (pre-update position). "
                  f"Skipping partial exit check.")
            continue

        df = get_stock_data(full_market_data, ticker)
        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        live_price = float(df[price_col].iloc[-1])
        gain_pct = (live_price - entry_price) / entry_price

        print(f" {ticker:18} | Entry: ₹{entry_price:.2f} | "
              f"Now: ₹{live_price:.2f} | Gain: {gain_pct*100:+.1f}%", end="")

        if gain_pct >= partial_exit_threshold and qty >= 2:
            sell_qty = max(1, int(qty * partial_exit_fraction))
            remain_qty = qty - sell_qty
            locked_gain = sell_qty * (live_price - entry_price)

            print(f" → 🔒 PARTIAL EXIT: Selling {sell_qty} of {qty} shares "
                  f"(locking ₹{locked_gain:,.0f} gain, {remain_qty} shares ride on)")

            record_transaction(
                ticker=ticker,
                side='sell',
                qty=sell_qty,
                price=live_price,
                strategy_name=f"Partial Exit +{gain_pct*100:.0f}%",
            )
            exits_executed += 1

        elif gain_pct >= partial_exit_threshold and qty < 2:
            print(f" → ℹ️ Gain ≥{partial_exit_threshold*100:.0f}% but only 1 share held — no split possible.")
        else:
            print()

    if exits_executed == 0:
        print(" No positions have reached the partial exit threshold yet.")

    return exits_executed

def check_stop_losses(tickers, full_market_data, hard_stop_pct=0.10):
    """
    Daily stop-loss check for ALL open positions.
    Triggers hard stop at -10% from entry_price.
    """
    print("\n--- PHASE 0.5: STOP-LOSS CHECK ---")
    
    portfolio = load_portfolio()
    holdings = portfolio.get('holdings', {})
    
    if not holdings:
        print(" No open positions.")
        return 0
    
    stops_triggered = 0
    
    for ticker, holding_data in list(holdings.items()):
        holding = _normalise_holding(holding_data)
        qty = holding['qty']
        entry_price = holding.get('entry_price', 0)
        
        if entry_price <= 0 or qty <= 0:
            continue
        
        df = get_stock_data(full_market_data, ticker)
        if df.empty:
            continue
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        current_price = float(df[price_col].iloc[-1])
        
        loss_pct = (current_price - entry_price) / entry_price
        
        print(f" {ticker:18} | Entry: ₹{entry_price:.2f} | Now: ₹{current_price:.2f} "
              f"| Loss: {loss_pct*100:+.1f}%", end="")
        
        if loss_pct <= -hard_stop_pct:
            print(f" → 🔴 STOP LOSS: Selling {qty} shares")
            record_transaction(ticker, 'sell', qty, current_price, 
                             f"Stop Loss ({loss_pct*100:.1f}%)")
            stops_triggered += 1
        else:
            print()
    
    if stops_triggered == 0:
        print(" No positions hit stop loss.")
    
    return stops_triggered
def run_autopilot_cycle(mode: str = "CONSERVATIVE"):
    print("\n" + "=" * 60)
    print(f"🤖 T_RAIDER AUTOPILOT — MODE: {mode}")
    print("=" * 60)

    portfolio = load_portfolio()
    tickers = get_config_tickers()
    # NEW:
    total_baseline_wealth = get_active_capital()

    print("\n📥 Fetching market data…")
    full_market_data = fetch_historical_data(tickers, period="1mo")

    # ── PHASE 0: PARTIAL EXITS ────────────────────────────────────────────
    check_partial_exits(tickers=tickers, full_market_data=full_market_data)
    
    # ── PHASE 0.5: STOP LOSSES ────────────────────────────────────────────
    check_stop_losses(tickers, full_market_data)

    # ── Get signals (pass mode to screener) ───────────────────────────────
    buy_signals, sell_signals = run_screener(tickers, mode=mode)

    # ── PHASE 1: FULL EXITS (strategy signal) ─────────────────────────────
    print("\n--- PHASE 1: EXITS ---")
    portfolio = load_portfolio()
    current_holdings = portfolio.get('holdings', {})

    for ticker, price in sell_signals:
        if ticker in current_holdings:
            holding = _normalise_holding(current_holdings[ticker])
            qty = holding['qty']
            print(f"🛑 EXIT: {ticker} — strategy signal flipped to SELL.")
            record_transaction(ticker, 'sell', qty, price, "Signal Exit")
            portfolio = load_portfolio()

    # ── PHASE 2: ENTRIES ──────────────────────────────────────────────────
    print("\n--- PHASE 2: VOLATILITY-ADJUSTED ENTRIES ---")

    for signal_data in buy_signals:
        portfolio = load_portfolio()

        ticker = signal_data['ticker']
        price = signal_data['price']

        holdings = portfolio.get('holdings', {})
        if ticker in holdings:
            holding = _normalise_holding(holdings[ticker])
            if holding['qty'] > 0:
                print(f" ⏭ {ticker} — already in portfolio ({holding['qty']} shares).")
                continue

        # ATR sizing
        df = get_stock_data(full_market_data, ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty or len(df) < 15:
            print(f"⚠️ SKIPPED: {ticker} (Insufficient data for ATR)")
            continue

        atr = calculate_atr(df)

        risk_per_trade = 0.01
        risk_per_share = atr * 2
        rupee_risk_allowed = total_baseline_wealth * risk_per_trade
        target_qty = int(rupee_risk_allowed // risk_per_share)

        max_position_cost = total_baseline_wealth * 0.20
        capped_qty = int(max_position_cost // price)

        final_qty = min(target_qty, capped_qty)
        total_cost = final_qty * price

        if total_cost > portfolio['cash']:
            final_qty = int(portfolio['cash'] // price)
            total_cost = final_qty * price
            if final_qty <= 0:
                print(f"⚠️ SKIPPED: {ticker} (Insufficient cash: ₹{portfolio['cash']:.2f})")
                continue

        if final_qty > 0:
            print(f"🚀 BUY: {ticker} | ₹{price:.2f} | ATR: {atr:.2f} | "
                  f"{final_qty} shares | Cost: ₹{total_cost:,.2f}")
            record_transaction(
                ticker, 'buy', final_qty, price,
                f"ATR Sized (ATR:{atr:.1f}) [{mode}]"
            )

    # ── Summary ───────────────────────────────────────────────────────────
    portfolio = load_portfolio()
    print("\n" + "=" * 60)
    print("✅ CYCLE COMPLETE")
    print(f" Mode: {mode}")
    print(f" Cash remaining: ₹{portfolio['cash']:,.2f}")
    print(f" Open positions: {len(portfolio.get('holdings', {}))}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='T_Raider Configurable Bot')
    parser.add_argument('--mode', choices=['CONSERVATIVE', 'BALANCED', 'AGGRESSIVE'],
                       default='CONSERVATIVE', help='Risk profile mode')
    args = parser.parse_args()
    run_autopilot_cycle(mode=args.mode)