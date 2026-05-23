"""
autopilot/auto_mode.py
──────────────────────
Auto-selects trading mode based on recent portfolio performance.
NO CONFIG FILE NEEDED. All thresholds are hardcoded constants.

Usage:
    from autopilot.auto_mode import auto_select_mode
    mode, reason = auto_select_mode()
"""

import json
import sys
import os
from pathlib import Path
from typing import Tuple, List, Dict

# ── Path fix for imports ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS — Hardcoded. Edit code to change thresholds.
# ═════════════════════════════════════════════════════════════════════════════

MIN_TRADES_FOR_AUTO = 10          # Minimum closed trades before auto-mode activates
WIN_RATE_AGGRESSIVE = 0.70        # 70%+ win rate for AGGRESSIVE
WIN_RATE_BALANCED = 0.55          # 55%+ win rate for BALANCED
MAX_DD_AGGRESSIVE = 0.05          # Max 5% drawdown for AGGRESSIVE
MAX_DD_BALANCED = 0.10            # Max 10% drawdown for BALANCED
MIN_SHARPE_AGGRESSIVE = 0.5       # Sharpe > 0.5 for AGGRESSIVE
LOOKBACK_TRADES = 20              # Win rate calculated over last 20 trades
LOOKBACK_EQUITY = 60              # Drawdown calculated over last 60 trades
DEFAULT_START_CAPITAL = 100000.0  # Original capital for equity curve


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO LOADER
# ═════════════════════════════════════════════════════════════════════════════

def load_portfolio() -> Dict:
    """Load portfolio.json with safe defaults."""
    path = Path("config/portfolio.json")
    if not path.exists():
        return {"history": [], "cash": DEFAULT_START_CAPITAL, "holdings": {}}
    
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"WARNING: Could not load portfolio.json ({e}). Using defaults.")
        return {"history": [], "cash": DEFAULT_START_CAPITAL, "holdings": {}}


# ═════════════════════════════════════════════════════════════════════════════
# P&L CALCULATION FROM BUY/SELL HISTORY
# ═════════════════════════════════════════════════════════════════════════════

def calculate_trade_pnls(history: List[Dict]) -> List[float]:
    """
    Calculate realized P&L from buy/sell history using FIFO matching.
    Handles partial sells and multiple positions per ticker.
    
    Expected history entry format:
        {"timestamp": "...", "ticker": "RELIANCE.NS", "side": "buy", "qty": 10, "price": 2400.0}
    """
    if not history:
        return []
    
    # Sort by timestamp (chronological)
    sorted_history = sorted(history, key=lambda x: x.get("timestamp", ""))
    
    # Track open positions: ticker -> list of {"qty": int, "price": float}
    positions: Dict[str, List[Dict]] = {}
    trade_pnls: List[float] = []
    
    for entry in sorted_history:
        ticker = entry.get("ticker", "")
        side = str(entry.get("side", "")).lower().strip()
        
        # Validate numeric fields
        try:
            qty = int(entry.get("qty", 0))
            price = float(entry.get("price", 0))
        except (ValueError, TypeError):
            continue  # Skip malformed entries
        
        if not ticker or qty <= 0 or price <= 0:
            continue
        
        if side == "buy":
            if ticker not in positions:
                positions[ticker] = []
            positions[ticker].append({"qty": qty, "price": price})
        
        elif side == "sell":
            if ticker not in positions or not positions[ticker]:
                continue  # Sell without matching buy (shouldn't happen)
            
            sell_qty = qty
            sell_price = price
            realized_pnl = 0.0
            
            # FIFO: match sells against earliest buys
            while sell_qty > 0 and positions[ticker]:
                buy = positions[ticker][0]
                match_qty = min(sell_qty, buy["qty"])
                
                # P&L = (sell_price - buy_price) * matched_qty
                pnl = (sell_price - buy["price"]) * match_qty
                realized_pnl += pnl
                
                buy["qty"] -= match_qty
                sell_qty -= match_qty
                
                if buy["qty"] <= 0:
                    positions[ticker].pop(0)
            
            trade_pnls.append(realized_pnl)
    
    return trade_pnls


# ═════════════════════════════════════════════════════════════════════════════
# METRIC CALCULATIONS
# ═════════════════════════════════════════════════════════════════════════════

def calculate_equity_curve(pnls: List[float], start_capital: float = DEFAULT_START_CAPITAL) -> List[float]:
    """Build equity curve from realized P&Ls."""
    equity = [start_capital]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)
    return equity


def calculate_max_drawdown(equity: List[float]) -> float:
    """Calculate maximum drawdown from peak."""
    if not equity or len(equity) < 2:
        return 0.0
    
    peak = equity[0]
    max_dd = 0.0
    
    for value in equity:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    
    return max_dd


def calculate_win_rate(pnls: List[float]) -> float:
    """Calculate win rate over last N trades."""
    if len(pnls) < 5:
        return 0.0
    
    recent = pnls[-LOOKBACK_TRADES:] if len(pnls) >= LOOKBACK_TRADES else pnls
    wins = sum(1 for p in recent if p > 0)
    return wins / len(recent)


def calculate_sharpe(pnls: List[float]) -> float:
    """Calculate approximate Sharpe ratio over last N trades."""
    if len(pnls) < 10:
        return 0.0
    
    recent = pnls[-LOOKBACK_EQUITY:] if len(pnls) >= LOOKBACK_EQUITY else pnls
    if not recent:
        return 0.0
    
    avg = sum(recent) / len(recent)
    variance = sum((p - avg) ** 2 for p in recent) / len(recent)
    std = variance ** 0.5
    
    return avg / std if std > 0 else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# MODE SELECTION LOGIC
# ═════════════════════════════════════════════════════════════════════════════

def auto_select_mode() -> Tuple[str, str]:
    """
    Auto-select trading mode based on portfolio performance.
    
    Returns:
        (mode, reason) where mode is CONSERVATIVE/BALANCED/AGGRESSIVE
    """
    portfolio = load_portfolio()
    history = portfolio.get("history", [])
    
    # Calculate realized P&Ls from buy/sell pairs
    trade_pnls = calculate_trade_pnls(history)
    
    # NEW USER: Not enough closed trades
    if len(trade_pnls) < MIN_TRADES_FOR_AUTO:
        return "CONSERVATIVE", (
            f"New user: {len(trade_pnls)}/{MIN_TRADES_FOR_AUTO} closed trades. "
            f"Defaulting to CONSERVATIVE."
        )
    
    # Calculate metrics
    win_rate = calculate_win_rate(trade_pnls)
    equity = calculate_equity_curve(trade_pnls[-LOOKBACK_EQUITY:])
    max_dd = calculate_max_drawdown(equity)
    sharpe = calculate_sharpe(trade_pnls)
    
    # ── AGGRESSIVE: All three conditions must be met ──────────────────────
    if (win_rate >= WIN_RATE_AGGRESSIVE and 
        max_dd < MAX_DD_AGGRESSIVE and 
        sharpe > MIN_SHARPE_AGGRESSIVE):
        return "AGGRESSIVE", (
            f"WR={win_rate*100:.0f}% (>={WIN_RATE_AGGRESSIVE*100:.0f}%), "
            f"DD={max_dd*100:.1f}% (<{MAX_DD_AGGRESSIVE*100:.0f}%), "
            f"Sharpe={sharpe:.2f} (>{MIN_SHARPE_AGGRESSIVE}). "
            f"AGGRESSIVE approved."
        )
    
    # ── BALANCED: Win rate and drawdown acceptable ────────────────────────
    if win_rate >= WIN_RATE_BALANCED and max_dd < MAX_DD_BALANCED:
        return "BALANCED", (
            f"WR={win_rate*100:.0f}% (>={WIN_RATE_BALANCED*100:.0f}%), "
            f"DD={max_dd*100:.1f}% (<{MAX_DD_BALANCED*100:.0f}%), "
            f"Sharpe={sharpe:.2f}. BALANCED."
        )
    
    # ── CONSERVATIVE: Identify actual failing metrics ─────────────────────
    failures = []
    if win_rate < WIN_RATE_BALANCED:
        failures.append(f"WR={win_rate*100:.0f}% (<{WIN_RATE_BALANCED*100:.0f}%)")
    if max_dd >= MAX_DD_BALANCED:
        failures.append(f"DD={max_dd*100:.1f}% (>={MAX_DD_BALANCED*100:.0f}%)")
    if sharpe <= 0:
        failures.append(f"Sharpe={sharpe:.2f} (<=0)")
    
    reason = " | ".join(failures) if failures else "Metrics below BALANCED thresholds"
    return "CONSERVATIVE", f"{reason}. CONSERVATIVE for capital protection."


# ═════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC (run standalone to test)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    portfolio = load_portfolio()
    history = portfolio.get("history", [])
    pnls = calculate_trade_pnls(history)
    
    print("=" * 60)
    print("AUTO MODE DIAGNOSTIC")
    print("=" * 60)
    print(f"Total history entries: {len(history)}")
    print(f"Closed trades (sell with matched buy): {len(pnls)}")
    print(f"Total realized P&L: Rs.{sum(pnls):,.2f}")
    
    if pnls:
        print(f"Winning trades: {sum(1 for p in pnls if p > 0)}")
        print(f"Losing trades: {sum(1 for p in pnls if p < 0)}")
        print(f"Average P&L: Rs.{sum(pnls)/len(pnls):,.2f}")
        print(f"Last 5 P&Ls: {[round(p, 2) for p in pnls[-5:]]}")
    
    print("-" * 60)
    mode, reason = auto_select_mode()
    print(f"Auto-selected mode: {mode}")
    print(f"Reason: {reason}")
    print("=" * 60)