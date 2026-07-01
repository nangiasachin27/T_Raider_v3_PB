import os
import sys
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import json
from pathlib import Path
from datetime import datetime, date

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autopilot.kelly_position_sizing import KellyPositionSizer

from daily_screener import run_screener
from autopilot.logger import (
    load_portfolio, record_transaction, save_portfolio,
    _normalise_holding, get_holding_entry_price
)
from utils import get_config_tickers
from ingestion.data_ingestion import fetch_historical_data, get_stock_data


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVE CONFIG — EDIT THESE FOR YOUR RISK APPETITE
# ═══════════════════════════════════════════════════════════════════════════════

ACTIVE_CONFIG = {
    "profit_target_pct": None,           # None = read from quarterly_config.json
    "target_from_expected_return": True, # Use optimal_params expected_return * 0.5
    "target_cap_min": 0.15,              # Minimum per-stock target +15%
    "target_cap_max": 0.50,              # Maximum per-stock target +50%

    "trailing_stop_pct": 0.15,           # Baseline trailing stop from peak
    "tighten_threshold_1": 0.50,         # At 50% of target, tighten to...
    "tightened_stop_1": 0.10,            # ...10%
    "tighten_threshold_2": 0.80,         # At 80% of target, tighten to...
    "tightened_stop_2": 0.05,            # ...5%

    #"dead_money_max_days": 60,           # Max hold without meaningful progress
    "dead_money_min_gain": 0.02,         # Must be +2% after dead_money_max_days

    "rotation_threshold": 20,              # New signal must score holding_score + 20
    "rotation_max_positions": 10,         # Target max open positions
    "rotation_min_cash_buffer": 5000,    # Always keep ₹5,000 cash

    "partial_exit_threshold": 0.20,        # +20% partial book (legacy)
    "partial_exit_fraction": 0.50,       # Sell 50% at partial
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_active_capital():
    override = Path("config/capital_override.json")
    if override.exists():
        with open(override) as f:
            return json.load(f).get("total_baseline_wealth", 100000.0)
    qcfg = _load_quarterly_config()
    return float(qcfg.get("current_base_capital", 100000.0))


def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_pc = np.abs(df['High'] - df['Close'].shift(1))
    low_pc = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0


def _load_quarterly_config() -> dict:
    path = Path("config/quarterly_config.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_optimal_params() -> dict:
    path = Path("config/optimal_params.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_stocks_universe() -> list:
    path = Path("config/stocks.json")
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("nifty_50", [])


def _get_target_pct(ticker: str, expected_return: float = 0.0) -> float:
    """Determine per-stock profit target from optimal_params or quarterly config."""
    cfg = ACTIVE_CONFIG
    qcfg = _load_quarterly_config()
    optimal = _load_optimal_params()

    # Priority 1: quarterly_config profit_target_pct as default
    default_target = qcfg.get("profit_target_pct")
    if default_target is not None:
        default_target = float(default_target)
    else:
        default_target = 0.30

    # Priority 2: per-ticker override from optimal_params expected_return
    plan = optimal.get(ticker, {})
    expected = plan.get("expected_return", expected_return)

    if expected > 0 and cfg["target_from_expected_return"]:
        target = expected / 100 * 0.5  # Use 50% of expected return as target
    else:
        target = default_target

    return float(np.clip(target, cfg["target_cap_min"], cfg["target_cap_max"]))


def _days_held(entry_date_str: str) -> int:
    if entry_date_str in (None, "", "unknown"):
        return 999
    try:
        entry = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
        return (date.today() - entry).days
    except Exception:
        return 999


def _get_live_trailing_stop_pct(gain_pct: float, target_pct: float) -> float:
    cfg = ACTIVE_CONFIG
    if target_pct <= 0:
        return cfg["trailing_stop_pct"]

    progress = gain_pct / target_pct
    if progress >= cfg["tighten_threshold_2"]:
        return cfg["tightened_stop_2"]
    elif progress >= cfg["tighten_threshold_1"]:
        return cfg["tightened_stop_1"]
    return cfg["trailing_stop_pct"]


def _read_active_risk_mult() -> float:
    path = Path("config/.active_risk_mult.json")
    if path.exists():
        with open(path) as f:
            return json.load(f).get("risk_multiplier", 1.0)
    return 1.0


def _clear_active_risk_mult():
    path = Path("config/.active_risk_mult.json")
    if path.exists():
        path.unlink()


def _score_opportunity(ticker: str, optimal_params: dict) -> float:
    """Score a potential buy from full universe. Higher = more attractive."""
    plan = optimal_params.get(ticker, {})
    expected = plan.get("expected_return", 0)
    stability = plan.get("stability_score", 0)
    sharpe = plan.get("sharpe_ratio", 0)
    composite = plan.get("composite_score", 0)
    max_dd = plan.get("max_drawdown", 100)
    strategy = plan.get("strategy", "NONE")

    if strategy == "NONE" or expected <= 0:
        return -999

    return (
        expected * 0.35 +
        stability * 0.25 +
        sharpe * 5 * 0.20 +
        composite * 100 * 0.15 +
        (100 - max_dd) * 0.05
    )


def _score_holding(ticker: str, holding: dict, live_price: float,
                   optimal_params: dict, days_held: int) -> float:
    """Score existing holding for rotation. Lower = weaker candidate for sale."""
    entry_price = holding.get("entry_price", 0)
    if entry_price <= 0:
        return -999

    gain_pct = (live_price - entry_price) / entry_price
    plan = optimal_params.get(ticker, {})
    expected = plan.get("expected_return", 0)
    stability = plan.get("stability_score", 0)

    score = (
        stability * 0.25 +
        expected * 0.25 +
        gain_pct * 100 * 0.30
    )

    cfg = ACTIVE_CONFIG

    qcfg = _load_quarterly_config()
    quarter_days = qcfg.get("quarter_days", 90)
    dynamic_dead_money_days = max(10, quarter_days // 3)
    if days_held > dynamic_dead_money_days and gain_pct < cfg["dead_money_min_gain"]:
        score -= 25
    if gain_pct < -0.05:
        score -= 15
    if gain_pct < -0.08:
        score -= 25
    if days_held > 30 and gain_pct < 0:
        score -= 10

    return score


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0: LIVE TRAILING STOPS
# ═══════════════════════════════════════════════════════════════════════════════
def check_trailing_stops(tickers, full_market_data):
    print(f"\n--- PHASE 0: VOLATILITY-ADAPTIVE TRAILING STOPS ---")
    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", {})
    if not holdings:
        print(" No open positions.")
        return 0

    stops_triggered = 0
    portfolio_changed = False 
    
    # Fetch broad market VIX to dynamically scale our ATR multipliers
    try:
        vix_series = yf.download("^INDIAVIX", period="2d", progress=False)["Close"].squeeze()
        latest_vix = float(vix_series.iloc[-1])
    except Exception:
        latest_vix = 15.0  # Safe default baseline VIX
        
    # Scale multiplier based on broad market volatility regime
    # High market fear = widen stops to survive noise; Low market fear = lock profits tighter
    if latest_vix > 18.0:
        atr_multiplier = 3.0  # High Volatility Regime
    elif latest_vix < 13.0:
        atr_multiplier = 1.75 # Low Volatility Regime
    else:
        atr_multiplier = 2.25 # Balanced Regime

    for ticker, holding_data in list(holdings.items()):
        holding = _normalise_holding(holding_data)
        qty = holding["qty"]
        entry_price = holding.get("entry_price", 0)
        peak_price = holding.get("peak_price", entry_price)

        if entry_price <= 0 or qty <= 0:
            continue

        df = get_stock_data(full_market_data, ticker)
        if df.empty:
            continue
            
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        current_price = float(df[price_col].iloc[-1])

        # Track and update peak prices
        new_peak = max(peak_price, current_price) if peak_price > 0 else current_price
        if new_peak != peak_price:
            holding["peak_price"] = round(new_peak, 4)
            portfolio["holdings"][ticker] = holding
            portfolio_changed = True

        # Calculate asset-specific ATR
        atr = calculate_atr(df, window=14)
        if atr <= 0:
            atr = current_price * 0.03  # Fallback to 3% price variance if ATR calculation fails

        # Volatility-adaptive trailing stop formulation
        vol_adaptive_stop_price = new_peak - (atr * atr_multiplier)
        
        # Protective emergency hard floor at 10% from initial entry point
        hard_floor_stop = entry_price * 0.90
        effective_stop = max(hard_floor_stop, vol_adaptive_stop_price)

        gain_pct = (current_price - entry_price) / entry_price
        print(f" {ticker:18} | Entry: ₹{entry_price:.2f} | "
              f"Peak: ₹{new_peak:.2f} | Now: ₹{current_price:.2f} | "
              f"Gain: {gain_pct*100:+.1f}% | ATR Mult: {atr_multiplier}x | "
              f"Stop: ₹{effective_stop:.2f}", end="")

        if current_price <= effective_stop:
            stop_type = "HARD FLOOR STOP" if hard_floor_stop > vol_adaptive_stop_price else "VOL-ADAPTIVE TRAIL"
            print(f" → 🔴 {stop_type} TRIGGERED: Selling {qty} shares")
            record_transaction(ticker, "sell", qty, current_price,
                               f"Adaptive Stop ({gain_pct*100:.1f}%, type={stop_type})")
            stops_triggered += 1
            # Remove from local track mapping immediately to allow other processes to allocate capital
            if ticker in portfolio["holdings"]:
                del portfolio["holdings"][ticker]
                portfolio_changed = True
        else:
            print()
            
    if portfolio_changed:
        save_portfolio(portfolio) 
        
    if stops_triggered == 0:
        print(" No positions hit volatility-adaptive trailing stops.")
    return stops_triggered
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0.1: PROFIT TARGETS (Active Full Exit per stock)
# ═══════════════════════════════════════════════════════════════════════════════

def check_profit_targets(tickers, full_market_data, optimized_params: dict):
    print(f"\n--- PHASE 0.1: PROFIT TARGETS (Active Push per Stock) ---")
    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", {})
    if not holdings:
        print(" No open positions.")
        return 0

    targets_hit = 0
    for ticker, holding_data in list(holdings.items()):
        holding = _normalise_holding(holding_data)
        qty = holding["qty"]
        entry_price = holding.get("entry_price", 0)
        if entry_price <= 0 or qty <= 0:
            continue

        df = get_stock_data(full_market_data, ticker)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        current_price = float(df[price_col].iloc[-1])
        gain_pct = (current_price - entry_price) / entry_price

        plan = optimized_params.get(ticker, {})
        expected_return = plan.get("expected_return", 0)
        target_pct = _get_target_pct(ticker, expected_return)

        print(f" {ticker:18} | Entry: ₹{entry_price:.2f} | "
              f"Now: ₹{current_price:.2f} | Gain: {gain_pct*100:+.1f}% | "
              f"Target: {target_pct*100:.1f}%", end="")

        if gain_pct >= target_pct:
            print(f" → 🎯 TARGET HIT! Selling {qty} shares")
            record_transaction(ticker, "sell", qty, current_price,
                               f"Profit Target ({gain_pct*100:.1f}% >= {target_pct*100:.1f}%)")
            targets_hit += 1
        else:
            print(f" → {(target_pct - gain_pct)*100:.1f}% to target")

    if targets_hit == 0:
        print(" No positions have reached their profit target yet.")
    return targets_hit


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0.2: DEAD MONEY / TIMEFRAME EXITS
# ═══════════════════════════════════════════════════════════════════════════════

def check_dead_money_exits(tickers, full_market_data):
    print(f"\n--- PHASE 0.2: DEAD MONEY / TIMEFRAME EXIT ---")
    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", {})
    if not holdings:
        print(" No open positions.")
        return 0

    exits_triggered = 0
    cfg = ACTIVE_CONFIG
    qcfg = _load_quarterly_config()
    quarter_days = qcfg.get("quarter_days", 90)
    dynamic_dead_money_days = max(10, quarter_days // 3)

    for ticker, holding_data in list(holdings.items()):
        holding = _normalise_holding(holding_data)
        qty = holding["qty"]
        entry_price = holding.get("entry_price", 0)
        entry_date = holding.get("entry_date", "unknown")
        if entry_price <= 0 or qty <= 0:
            continue

        days = _days_held(entry_date)
        if days < dynamic_dead_money_days:
            continue

        df = get_stock_data(full_market_data, ticker)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        current_price = float(df[price_col].iloc[-1])
        gain_pct = (current_price - entry_price) / entry_price

        print(f" {ticker:18} | Held: {days}d | Gain: {gain_pct*100:+.1f}%", end="")

        if gain_pct < cfg["dead_money_min_gain"]:
            print(f" → ⏳ DEAD MONEY: Held {days}d, only {gain_pct*100:+.1f}%. "
                  f"Selling {qty} shares to free capital.")
            record_transaction(ticker, "sell", qty, current_price,
                               f"Dead Money Exit ({days}d, {gain_pct*100:.1f}%)")
            exits_triggered += 1
        else:
            print(f" → ✅ Held {days}d but gain {gain_pct*100:+.1f}% — keeping.")

    if exits_triggered == 0:
        print(" No dead-money positions found.")
    return exits_triggered


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0.3: PARTIAL EXITS (Legacy)
# ═══════════════════════════════════════════════════════════════════════════════

def check_partial_exits(tickers, full_market_data,
                        partial_exit_threshold=0.20,
                        partial_exit_fraction=0.50):
    print(f"\n--- PHASE 0.3: PARTIAL EXITS (Profit Lock-In at +{partial_exit_threshold*100:.0f}%) ---")
    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", {})
    if not holdings:
        print(" No open positions to check.")
        return 0

    exits_executed = 0
    for ticker, holding_data in list(holdings.items()):
        holding = _normalise_holding(holding_data)
        qty = holding["qty"]
        entry_price = holding["entry_price"]

        if entry_price <= 0:
            print(f" ⚠️ {ticker} — entry price unknown. Skipping partial exit check.")
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


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0.5: HARD STOP LOSSES
# ═══════════════════════════════════════════════════════════════════════════════

def check_stop_losses(tickers, full_market_data, hard_stop_pct=0.10):
    print("\n--- PHASE 0.5: STOP-LOSS CHECK ---")
    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", {})
    if not holdings:
        print(" No open positions.")
        return 0

    stops_triggered = 0
    for ticker, holding_data in list(holdings.items()):
        holding = _normalise_holding(holding_data)
        qty = holding["qty"]
        entry_price = holding.get("entry_price", 0)
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


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1.5: CAPITAL ROTATION (Enhanced v2)
# ═══════════════════════════════════════════════════════════════════════════════

def rotate_capital_for_buy(buy_signals, full_market_data,
                           portfolio, total_baseline_wealth,
                           optimized_params: dict,
                           mode: str = "CONSERVATIVE"):
    """
    Enhanced capital rotation:
    1. If cash insufficient for buy signal → sell weakest holding
    2. If too many positions → proactively sell weakest, buy best from universe
    3. Uses full universe scoring from optimal_params
    """
    print("\n--- PHASE 1.5: CAPITAL ROTATION ---")
    executed = []
    cash = portfolio.get("cash", 0)
    holdings = portfolio.get("holdings", {})
    cfg = ACTIVE_CONFIG
    universe = _load_stocks_universe()

    if not buy_signals and len(holdings) < cfg["rotation_max_positions"]:
        print(" No rotation needed — cash and position count OK.")
        return portfolio, executed

    # Score all buy signals
    scored_signals = []
    for sig in buy_signals:
        ticker = sig["ticker"]
        plan = optimized_params.get(ticker, {})
        score = _score_opportunity(ticker, optimized_params)
        scored_signals.append((score, sig, plan))
    scored_signals.sort(key=lambda x: x[0], reverse=True)

    # Score all current holdings
    holding_scores = []
    for hticker, hdata in list(holdings.items()):
        holding = _normalise_holding(hdata)
        if holding["qty"] <= 0:
            continue

        hdf = get_stock_data(full_market_data, hticker)
        if hdf.empty:
            continue
        if isinstance(hdf.columns, pd.MultiIndex):
            hdf.columns = hdf.columns.get_level_values(0)

        hprice_col = 'Adj Close' if 'Adj Close' in hdf.columns else 'Close'
        hprice = float(hdf[hprice_col].iloc[-1])
        days = _days_held(holding.get("entry_date", "unknown"))
        hscore = _score_holding(hticker, holding, hprice, optimized_params, days)
        holding_scores.append((hscore, hticker, holding, hprice, days))

    # Sort: weakest first
    holding_scores.sort(key=lambda x: x[0])

    # ── Case 1: Cash insufficient for buy signals ────────────
    for score, sig, plan in scored_signals:
        ticker = sig["ticker"]
        price = sig["price"]
        if ticker in holdings:
            continue

        df = get_stock_data(full_market_data, ticker)
        if df.empty or len(df) < 15:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        atr = calculate_atr(df)

            # Kelly sizing (matches daily_screener.py)
        target_qty, sizing_reason = KellyPositionSizer.calculate(
                ticker=ticker,
                portfolio=portfolio,
                optimal_params=optimized_params,
                atr=atr,
                current_price=price,
                mode=mode,
                capital=total_baseline_wealth
            )

            # Apply active risk multiplier if set by ActiveProfitEngine
        risk_mult = _read_active_risk_mult()
        if risk_mult != 1.0:
            target_qty = int(target_qty * risk_mult)

            # 20% concentration cap (kept from original bot logic)
        max_position_cost = total_baseline_wealth * 0.20
        capped_qty = int(max_position_cost // price) if price > 0 else 0
        final_qty = min(target_qty, capped_qty)
        total_cost = final_qty * price

        # NEW: Immediately skip if the stock is too risky/expensive to buy even 1 share
        if final_qty <= 0:
            print(f"\n ⚠️ SKIPPING {ticker}: Too volatile or expensive. Buying 1 share exceeds max risk limit.")
            continue

        if total_cost <= cash:
            continue  # Enough cash, no rotation needed

        print(f"\n 💡 Rotation needed for {ticker}: need ₹{total_cost:,.0f}, have ₹{cash:,.0f}")


        if not holding_scores:
            print("   No holdings to rotate.")
            continue

        weakest_score, weakest_ticker, weakest_holding, weakest_price, weakest_days = holding_scores[0]

        print(f"   Weakest holding: {weakest_ticker} (score={weakest_score:.1f}, held {weakest_days}d)")
        print(f"   New signal score: {score:.1f}")

        if score >= weakest_score + cfg["rotation_threshold"]:
            wqty = weakest_holding["qty"]
            proceeds = wqty * weakest_price
            print(f"   ✅ ROTATING: Sell {wqty} {weakest_ticker} @ ₹{weakest_price:.2f} "
                  f"(proceeds ₹{proceeds:,.0f}) → Buy {ticker}")
            record_transaction(weakest_ticker, "sell", wqty, weakest_price,
                               f"Capital Rotation → {ticker}")
            executed.append({
                "sold": weakest_ticker,
                "sold_qty": wqty,
                "sold_price": weakest_price,
                "bought": ticker,
                "bought_qty": final_qty,
                "bought_price": price,
                "reason": f"Score {score:.1f} vs {weakest_score:.1f}"
            })

            portfolio = load_portfolio()
            cash = portfolio.get("cash", 0)
            holdings = portfolio.get("holdings", {})

            if total_cost <= cash and final_qty > 0:
                print(f"   🚀 ROTATION BUY: {ticker} | ₹{price:.2f} | {final_qty} shares")
                record_transaction(ticker, "buy", final_qty, price,
                                   f"ATR Sized (Rotation from {weakest_ticker})")
                portfolio = load_portfolio()
                cash = portfolio.get("cash", 0)
                holdings = portfolio.get("holdings", {})
            else:
                print(f"   ⚠️ Still insufficient cash after rotation (₹{cash:,.0f} < ₹{total_cost:,.0f})")
        else:
            print(f"   ❌ No rotation: score gap {score - weakest_score:.1f} < threshold {cfg['rotation_threshold']}")

    # ── Case 2: Too many positions → proactive rebalancing ────
    if len(holdings) >= cfg["rotation_max_positions"]:
        print(f"\n 📊 Position limit reached ({len(holdings)}/{cfg['rotation_max_positions']}). "
              f"Evaluating proactive rebalancing...")

        if scored_signals and holding_scores:
            best_score, best_sig, best_plan = scored_signals[0] # Grab the #1 approved signal
            best_ticker = best_sig["ticker"]
            weakest_score, weakest_ticker, weakest_holding, weakest_price, weakest_days = holding_scores[0]

            if best_score >= weakest_score + cfg["rotation_threshold"]:
                wqty = weakest_holding["qty"]
                print(f"\n 🔄 PROACTIVE REBALANCE: {weakest_ticker} (score {weakest_score:.1f}) → "
                      f"{best_ticker} (score {best_score:.1f})")
                record_transaction(weakest_ticker, "sell", wqty, weakest_price,
                                   f"Proactive Rebalance → {best_ticker}")
                executed.append({
                    "sold": weakest_ticker,
                    "sold_qty": wqty,
                    "sold_price": weakest_price,
                    "bought": best_ticker,
                    "bought_qty": 0,  # Will be sized in entry phase
                    "bought_price": 0,
                    "reason": f"Proactive: {best_score:.1f} vs {weakest_score:.1f}"
                })

    if not executed:
        print(" No capital rotations executed today.")
    return portfolio, executed


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AUTOPILOT CYCLE
# ═══════════════════════════════════════════════════════════════════════════════

def run_autopilot_cycle(mode: str = "CONSERVATIVE", market: str = "INDIA", filters: str = "ALL"):
    print("\n" + "=" * 60)
    print(f"🤖 T_RAIDER AUTOPILOT v4 — MODE: {mode}")
    print("   Active Profit Pushing + Capital Rotation")
    print("=" * 60)

    # ── Market hours gate ─────────────────────────────────────────────
    
    from market_hours import guard_or_exit
    guard_or_exit(market, exit_on_closed=True)

    portfolio = load_portfolio()
    tickers = get_config_tickers()
    total_baseline_wealth = get_active_capital()

    # Load optimized params for targets and rotation scoring
    optimized_params = _load_optimal_params()

    print("\n📥 Fetching market data…")
    full_market_data = fetch_historical_data(tickers, period="1mo")

    # ── PHASE 0: TRAILING STOPS ─────────────────────────────────────────
    check_trailing_stops(tickers, full_market_data)

    # ── PHASE 0.1: PROFIT TARGETS ───────────────────────────────────────
    check_profit_targets(tickers, full_market_data, optimized_params)

    # ── PHASE 0.2: DEAD MONEY EXITS ───────────────────────────────────
    check_dead_money_exits(tickers, full_market_data)

    # ── PHASE 0.3: PARTIAL EXITS ──────────────────────────────────────
    check_partial_exits(tickers, full_market_data)

    # ── PHASE 0.5: STOP LOSSES ──────────────────────────────────────────
    check_stop_losses(tickers, full_market_data)

    # ── Get signals (pass mode to screener) ───────────────────────────────
    buy_signals, sell_signals = run_screener(tickers, mode=mode, filters=filters)

    # ── PHASE 1: FULL EXITS (strategy signal) ─────────────────────────
    print("\n--- PHASE 1: STRATEGY SIGNAL EXITS ---")
    portfolio = load_portfolio()
    current_holdings = portfolio.get("holdings", {})

    for ticker, price in sell_signals:
        if ticker in current_holdings:
            holding = _normalise_holding(current_holdings[ticker])
            qty = holding["qty"]
            print(f"🛑 EXIT: {ticker} — strategy signal flipped to SELL.")
            record_transaction(ticker, "sell", qty, price, "Signal Exit")
            portfolio = load_portfolio()

    # ── PHASE 1.5: CAPITAL ROTATION ───────────────────────────────────
    portfolio, rotations = rotate_capital_for_buy(
        buy_signals, full_market_data, portfolio,
        total_baseline_wealth, optimized_params,
        mode=mode
    )

    # ── PHASE 2: ENTRIES ──────────────────────────────────────────────
    print("\n--- PHASE 2: VOLATILITY-ADJUSTED ENTRIES ---")

    # CRITICAL FIX (Option A): Sort buy signals by their opportunity score descending.
    # This prevents lower-ranked assets (e.g., COALINDIA.NS) from front-running 
    # and consuming cash explicitly intended for high-alpha targets (e.g., INDIANB.NS).
    buy_signals = sorted(
        buy_signals, 
        key=lambda x: _score_opportunity(x["ticker"], optimized_params), 
        reverse=True
    )
    # Dynamic risk from ActiveProfitEngine
    risk_mult = _read_active_risk_mult()
    print(f"\n📐 Active Risk Multiplier: {risk_mult:.1f}x")

    for signal_data in buy_signals:
        portfolio = load_portfolio()
        ticker = signal_data["ticker"]
        price = signal_data["price"]

        holdings = portfolio.get("holdings", {})
        if ticker in holdings:
            holding = _normalise_holding(holdings[ticker])
            if holding["qty"] > 0:
                print(f" ⏭ {ticker} — already in portfolio ({holding['qty']} shares).")
                continue

        df = get_stock_data(full_market_data, ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 15:
            print(f"⚠️ SKIPPED: {ticker} (Insufficient data for ATR)")
            continue

        atr = calculate_atr(df)

        # Kelly sizing (matches daily_screener.py)
        target_qty, sizing_reason = KellyPositionSizer.calculate(
            ticker=ticker,
            portfolio=portfolio,
            optimal_params=optimized_params,
            atr=atr,
            current_price=price,
            mode=mode,
            capital=total_baseline_wealth
        )

        # Apply active risk multiplier if set by ActiveProfitEngine
        risk_mult = _read_active_risk_mult()
        if risk_mult != 1.0:
            target_qty = int(target_qty * risk_mult)

        # 20% concentration cap
        max_position_cost = total_baseline_wealth * 0.35
        capped_qty = int(max_position_cost // price) if price > 0 else 0
        final_qty = min(target_qty, capped_qty)
        total_cost = final_qty * price

        if total_cost > portfolio["cash"]:
            final_qty = int(portfolio["cash"] // price) if price > 0 else 0
            total_cost = final_qty * price
            if final_qty <= 0:
                print(f"⚠️ SKIPPED: {ticker} (Insufficient cash: ₹{portfolio['cash']:.2f})")
                continue

        if final_qty > 0:
            print(f"🚀 BUY: {ticker} | ₹{price:.2f} | ATR: {atr:.2f} | "
                  f"{final_qty} shares | Cost: ₹{total_cost:,.2f}")
            record_transaction(
                ticker, "buy", final_qty, price,
                f"ATR Sized (ATR:{atr:.1f}) [{mode}]"
            )

    # ── Summary ───────────────────────────────────────────────────────
    portfolio = load_portfolio()
    print("\n" + "=" * 60)
    print("✅ CYCLE COMPLETE")
    print(f" Mode: {mode}")
    print(f" Cash remaining: ₹{portfolio['cash']:,.2f}")
    print(f" Open positions: {len(portfolio.get('holdings', {}))}")
    if rotations:
        print(f" Capital rotations today: {len(rotations)}")
        for r in rotations:
            print(f"   {r['sold']} → {r['bought']} ({r['reason']})")
    print("=" * 60)

    _clear_active_risk_mult()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='T_Raider Active Bot v4')
    parser.add_argument('--mode', choices=['CONSERVATIVE', 'BALANCED', 'AGGRESSIVE'],
                        default='CONSERVATIVE', help='Risk profile mode')
    args = parser.parse_args()
    run_autopilot_cycle(mode=args.mode)