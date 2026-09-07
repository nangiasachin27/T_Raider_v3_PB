"""
autopilot/auto_mode.py
──────────────────────
Auto-selects trading mode based on recent portfolio performance.

Design: AGGRESSIVE-by-default with a downgrade-only circuit breaker.
    - Once past the new-user safety window, the assumed mode is AGGRESSIVE.
    - Trailing win-rate / drawdown / Sharpe / consecutive-loss checks can
      only pull the mode DOWN to BALANCED or CONSERVATIVE — they can never
      push it up. Recovering back toward AGGRESSIVE requires several
      consecutive clean reads (hysteresis), so one good day right after a
      downgrade doesn't immediately re-arm full size.
    - AGGRESSIVE is additionally capped to BALANCED whenever Nifty is below
      its 50-day EMA (regime gate), independent of the account's own trade
      history.

All thresholds live in config/auto_mode_config.json (falls back to the
defaults below if the file is missing/unreadable) — no code changes needed
to retune.

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
# CONFIG LOADING — config/auto_mode_config.json is the source of truth.
# The literals below are only a safety-net fallback if the file is missing.
# ═════════════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    "min_trades_for_auto": 10,
    "lookback_trades": 20,
    "lookback_equity": 60,
    "default_start_capital": 100000.0,

    # ── Downgrade-only circuit breaker thresholds ──────────────────────────
    # Default mode (once past the new-user window) is AGGRESSIVE. Any ONE
    # of the conditions below being breached pulls the mode down a tier.
    "downgrade_balanced_max_dd": 0.05,
    "downgrade_balanced_win_rate": 0.45,
    "downgrade_balanced_sharpe": 0.0,
    "downgrade_balanced_consecutive_losses": 3,

    "downgrade_conservative_max_dd": 0.10,
    "downgrade_conservative_win_rate": 0.35,
    "downgrade_conservative_sharpe": -0.5,
    "downgrade_conservative_consecutive_losses": 5,

    # ── Hysteresis: instant downgrades, delayed upgrades ───────────────────
    "hysteresis_confirmations": 2,

    # ── Regime gate: caps AGGRESSIVE when Nifty is below its 50-EMA ────────
    "regime_gate_enabled": True,
}


def _load_auto_mode_config() -> Dict:
    path = Path(PROJECT_ROOT) / "config" / "auto_mode_config.json"
    try:
        with open(path) as f:
            cfg = json.load(f)
        return {**_DEFAULTS, **cfg}
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        print(f"WARNING: Could not load config/auto_mode_config.json ({e}). Using built-in defaults.")
        return dict(_DEFAULTS)


_CFG = _load_auto_mode_config()

MIN_TRADES_FOR_AUTO = _CFG["min_trades_for_auto"]
LOOKBACK_TRADES = _CFG["lookback_trades"]
LOOKBACK_EQUITY = _CFG["lookback_equity"]
DEFAULT_START_CAPITAL = _CFG["default_start_capital"]


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


def calculate_consecutive_losses(pnls: List[float]) -> int:
    """
    Count the current trailing streak of non-winning closed trades
    (pnl <= 0), most-recent-first. Resets to 0 on the first winning trade
    encountered walking backward. This reacts much faster than the
    aggregate win-rate/drawdown/Sharpe metrics to a fresh losing streak,
    since those are averaged/lookback metrics that can stay within normal
    range even while several trades in a row have just lost.
    """
    streak = 0
    for p in reversed(pnls):
        if p <= 0:
            streak += 1
        else:
            break
    return streak


# ═════════════════════════════════════════════════════════════════════════════
# MODE RANKING + STATE PERSISTENCE (for hysteresis)
# ═════════════════════════════════════════════════════════════════════════════

MODE_RANK = {"CONSERVATIVE": 0, "BALANCED": 1, "AGGRESSIVE": 2}
STATE_PATH = Path(PROJECT_ROOT) / "config" / "auto_mode_state.json"


def _load_state() -> Dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return {"confirmed_mode": "AGGRESSIVE", "pending_mode": None, "pending_count": 0}


def _save_state(state: Dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except IOError as e:
        print(f"WARNING: Could not save {STATE_PATH} ({e}). Hysteresis state not persisted.")


# ═════════════════════════════════════════════════════════════════════════════
# RAW MODE SELECTION — AGGRESSIVE by default, downgrade-only circuit breaker
# ═════════════════════════════════════════════════════════════════════════════

def _raw_select_mode() -> Tuple[str, str]:
    """
    AGGRESSIVE-by-default mode selection. Once the account has enough closed
    trades to be past the new-user safety window, the assumed mode is
    AGGRESSIVE; trailing performance can only pull it DOWN a tier (or two),
    never up. This is the pre-hysteresis, pre-regime-gate signal — use
    auto_select_mode() for the final production decision.

    Returns:
        (mode, reason) where mode is CONSERVATIVE/BALANCED/AGGRESSIVE
    """
    portfolio = load_portfolio()
    history = portfolio.get("history", [])
    trade_pnls = calculate_trade_pnls(history)

    # NEW USER: not enough closed trades to trust any circuit breaker yet —
    # this safety net is unchanged from the old earn-your-way-up design.
    if len(trade_pnls) < MIN_TRADES_FOR_AUTO:
        return "CONSERVATIVE", (
            f"New user: {len(trade_pnls)}/{MIN_TRADES_FOR_AUTO} closed trades. "
            f"Defaulting to CONSERVATIVE until enough track record exists."
        )

    win_rate = calculate_win_rate(trade_pnls)
    equity = calculate_equity_curve(trade_pnls[-LOOKBACK_EQUITY:])
    max_dd = calculate_max_drawdown(equity)
    sharpe = calculate_sharpe(trade_pnls)
    consec_losses = calculate_consecutive_losses(trade_pnls)

    metrics_str = f"WR={win_rate*100:.0f}% DD={max_dd*100:.1f}% Sharpe={sharpe:.2f} ConsecLosses={consec_losses}"

    # ── Tier 2 breach: force all the way down to CONSERVATIVE ─────────────
    con_failures = []
    if max_dd >= _CFG["downgrade_conservative_max_dd"]:
        con_failures.append(f"DD={max_dd*100:.1f}% (>={_CFG['downgrade_conservative_max_dd']*100:.0f}%)")
    if win_rate < _CFG["downgrade_conservative_win_rate"]:
        con_failures.append(f"WR={win_rate*100:.0f}% (<{_CFG['downgrade_conservative_win_rate']*100:.0f}%)")
    if sharpe < _CFG["downgrade_conservative_sharpe"]:
        con_failures.append(f"Sharpe={sharpe:.2f} (<{_CFG['downgrade_conservative_sharpe']})")
    if consec_losses >= _CFG["downgrade_conservative_consecutive_losses"]:
        con_failures.append(f"ConsecLosses={consec_losses} (>={_CFG['downgrade_conservative_consecutive_losses']})")
    if con_failures:
        return "CONSERVATIVE", f"Circuit breaker (CONSERVATIVE): {' | '.join(con_failures)}. [{metrics_str}]"

    # ── Tier 1 breach: downgrade to BALANCED ───────────────────────────────
    bal_failures = []
    if max_dd >= _CFG["downgrade_balanced_max_dd"]:
        bal_failures.append(f"DD={max_dd*100:.1f}% (>={_CFG['downgrade_balanced_max_dd']*100:.0f}%)")
    if win_rate < _CFG["downgrade_balanced_win_rate"]:
        bal_failures.append(f"WR={win_rate*100:.0f}% (<{_CFG['downgrade_balanced_win_rate']*100:.0f}%)")
    if sharpe < _CFG["downgrade_balanced_sharpe"]:
        bal_failures.append(f"Sharpe={sharpe:.2f} (<{_CFG['downgrade_balanced_sharpe']})")
    if consec_losses >= _CFG["downgrade_balanced_consecutive_losses"]:
        bal_failures.append(f"ConsecLosses={consec_losses} (>={_CFG['downgrade_balanced_consecutive_losses']})")
    if bal_failures:
        return "BALANCED", f"Circuit breaker (BALANCED): {' | '.join(bal_failures)}. [{metrics_str}]"

    # ── No breach: stay at the AGGRESSIVE default ──────────────────────────
    return "AGGRESSIVE", f"All circuit breakers clear. [{metrics_str}]"


# ═════════════════════════════════════════════════════════════════════════════
# REGIME GATE — caps AGGRESSIVE when the broad market is in a downtrend
# ═════════════════════════════════════════════════════════════════════════════

def _apply_regime_gate(mode: str, reason: str) -> Tuple[str, str]:
    """
    If regime_gate_enabled and mode == AGGRESSIVE, cap to BALANCED whenever
    Nifty 50 is below its 50-day EMA. Reuses the same regime check already
    used to gate BUY signals in daily_screener.py, so this stays consistent
    with the rest of the system rather than introducing a second definition
    of "downtrend". Fails open (no cap) if the regime check itself fails.
    """
    if mode != "AGGRESSIVE" or not _CFG.get("regime_gate_enabled", True):
        return mode, reason

    try:
        from ingestion.nse_constituents import get_market_regime, regime_summary
        is_uptrend, nifty_close, nifty_ema = get_market_regime()
    except Exception as e:
        print(f"WARNING: Regime gate check failed ({e}). Not capping mode.")
        return mode, reason

    if not is_uptrend:
        summary = regime_summary(is_uptrend, nifty_close, nifty_ema)
        return "BALANCED", (
            f"{reason} | Regime gate: Nifty in DOWNTREND ({summary}) — "
            f"capped AGGRESSIVE to BALANCED."
        )
    return mode, reason


# ═════════════════════════════════════════════════════════════════════════════
# HYSTERESIS — instant downgrades, delayed upgrades
# ═════════════════════════════════════════════════════════════════════════════

def _apply_hysteresis(raw_mode: str, raw_reason: str) -> Tuple[str, str]:
    """
    De-risking (moving to a LESS aggressive mode) always applies immediately
    — the whole point of a circuit breaker is to cut risk fast. Recovering
    back toward AGGRESSIVE requires `hysteresis_confirmations` consecutive
    raw evaluations agreeing that all breakers are clear, so a single clean
    day right after a downgrade doesn't instantly re-arm full size.
    """
    confirmations_needed = int(_CFG.get("hysteresis_confirmations", 2))
    state = _load_state()
    confirmed = state.get("confirmed_mode", "AGGRESSIVE")

    if MODE_RANK[raw_mode] <= MODE_RANK[confirmed]:
        # Downgrade or unchanged: apply immediately, clear any pending upgrade.
        new_state = {"confirmed_mode": raw_mode, "pending_mode": None, "pending_count": 0}
        _save_state(new_state)
        if raw_mode != confirmed:
            raw_reason = f"{raw_reason} | De-risked immediately from {confirmed} (no hysteresis on downgrades)."
        return raw_mode, raw_reason

    # Upgrade requested — needs confirmations_needed consecutive agreeing reads.
    if state.get("pending_mode") == raw_mode:
        pending_count = state.get("pending_count", 0) + 1
    else:
        pending_count = 1

    if pending_count >= confirmations_needed:
        new_state = {"confirmed_mode": raw_mode, "pending_mode": None, "pending_count": 0}
        _save_state(new_state)
        return raw_mode, f"{raw_reason} | Recovery confirmed after {pending_count}/{confirmations_needed} clean reads."

    new_state = {"confirmed_mode": confirmed, "pending_mode": raw_mode, "pending_count": pending_count}
    _save_state(new_state)
    return confirmed, (
        f"{raw_reason} | Recovery to {raw_mode} pending ({pending_count}/{confirmations_needed} "
        f"clean reads) — staying at {confirmed} for now."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def auto_select_mode() -> Tuple[str, str]:
    """
    Production mode decision: AGGRESSIVE-by-default circuit breaker →
    hysteresis (instant downgrades, delayed recovery) → regime gate (caps
    AGGRESSIVE in a Nifty downtrend). Call this from quarterly_manager.py /
    profit_chaser.py.

    Returns:
        (mode, reason) where mode is CONSERVATIVE/BALANCED/AGGRESSIVE
    """
    raw_mode, raw_reason = _raw_select_mode()
    mode, reason = _apply_hysteresis(raw_mode, raw_reason)
    mode, reason = _apply_regime_gate(mode, reason)
    return mode, reason


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