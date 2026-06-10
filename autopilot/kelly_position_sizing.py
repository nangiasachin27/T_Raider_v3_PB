"""
Kelly Position Sizing Module — Dynamic risk allocation
─────────────────────────────────────────────────────
Replaces fixed 1% risk with Kelly Criterion-based sizing.
Scales position size based on proven edge (win rate, avg win, avg loss).

Usage:
    from daily_screener import KellyPositionSizer
    qty, reason = KellyPositionSizer.calculate(ticker, portfolio, optimal_params, atr, price)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

class KellyPositionSizer:
    """
    Implements fractional Kelly position sizing.

    Kelly fraction = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
    We use "quarter Kelly" (0.25 * Kelly) for safety.
    """

    # Hardcoded safety limits
    KELLY_FRACTION = 0.50        # Quarter Kelly (conservative)
    MAX_RISK_PER_TRADE = 0.04    # Hard cap: 4% of capital max
    MIN_RISK_PER_TRADE = 0.005   # Floor: 0.5% of capital min
    MIN_TRADES_FOR_KELLY = 10    # Need 10+ trades before Kelly activates
    ATR_MULTIPLIER = 2.0         # Risk per share = ATR * 2

    @classmethod
    def calculate(
        cls,
        ticker: str,
        portfolio: Dict,
        optimal_params: Dict,
        atr: float,
        current_price: float,
        mode: str = "CONSERVATIVE",
        capital: Optional[float] = None
    ) -> Tuple[int, str]:
        """
        Calculate position size using Kelly Criterion.

        Returns:
            (quantity: int, reason: str)
        """
        capital = capital if capital is not None else portfolio.get("cash", 100000.0)

        # Validate inputs
        if atr <= 0 or current_price <= 0:
            return 0, "Invalid ATR or price"

        # Get historical P&Ls for this ticker
        ticker_pnls = cls._get_ticker_pnls(ticker, portfolio)

        # Not enough history — fall back to fixed sizing
        if len(ticker_pnls) < cls.MIN_TRADES_FOR_KELLY:
            qty, reason = cls._fixed_sizing(capital, atr, mode)
            return qty, f"Kelly: {len(ticker_pnls)}/{cls.MIN_TRADES_FOR_KELLY} trades. {reason}"

        # Calculate Kelly components
        wins = [p for p in ticker_pnls if p > 0]
        losses = [p for p in ticker_pnls if p < 0]

        if not wins or not losses:
            qty, reason = cls._fixed_sizing(capital, atr, mode)
            return qty, f"Kelly: No wins/losses mix. {reason}"

        win_rate = len(wins) / len(ticker_pnls)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        # Edge case: avg_win == 0
        if avg_win == 0:
            qty, reason = cls._fixed_sizing(capital, atr, mode)
            return qty, f"Kelly: Zero avg win. {reason}"

        # Kelly formula: f* = (p*b - q) / b
        # Where p = win_rate, q = loss_rate, b = avg_win/avg_loss (payoff ratio)
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # Standard Kelly: (win_rate * payoff_ratio - loss_rate) / payoff_ratio
        # Simplified: (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        loss_rate = 1 - win_rate
        kelly_raw = (win_rate * avg_win - loss_rate * avg_loss) / avg_win

        # Apply fractional Kelly and bounds
        kelly_adjusted = kelly_raw * cls.KELLY_FRACTION

        # Mode-based risk scaling
        mode_multiplier = {"CONSERVATIVE": 0.5, "BALANCED": 1.0, "AGGRESSIVE": 1.5}.get(mode, 1.0)

        risk_fraction = kelly_adjusted * mode_multiplier
        risk_fraction = max(cls.MIN_RISK_PER_TRADE, min(risk_fraction, cls.MAX_RISK_PER_TRADE))

        # Calculate rupee risk and position size
        rupee_risk = capital * risk_fraction
        risk_per_share = atr * cls.ATR_MULTIPLIER

        target_qty = int(rupee_risk // risk_per_share)

        # Ensure we don't exceed available cash
        max_affordable_qty = int(capital // current_price)
        target_qty = min(target_qty, max_affordable_qty)

        return target_qty, (
            f"Kelly={kelly_raw:.3f}, Q-Kelly={kelly_adjusted:.3f}, "
            f"Risk={risk_fraction*100:.2f}%, "
            f"WR={win_rate*100:.0f}%, Payoff={payoff_ratio:.2f}x, "
            f"Qty={target_qty}"
        )

    @classmethod
    def _get_ticker_pnls(cls, ticker: str, portfolio: Dict) -> list:
        """Extract realized P&Ls for a specific ticker."""
        history = portfolio.get("history", [])
        pnls = []

        for entry in history:
            if (entry.get("ticker") == ticker and 
                entry.get("side", "").lower() == "sell" and
                "pnl" in entry):
                pnls.append(float(entry["pnl"]))

        # Fallback: calculate from price/qty if pnl not stored
        if not pnls:
            pnls = cls._calculate_pnls_from_history(ticker, history)

        return pnls

    @classmethod
    def _calculate_pnls_from_history(cls, ticker: str, history: list) -> list:
        """Calculate P&Ls from buy/sell history if 'pnl' field missing."""
        ticker_trades = [t for t in history if t.get("ticker") == ticker]
        ticker_trades.sort(key=lambda x: x.get("timestamp", ""))

        positions = []  # FIFO queue of (qty, price)
        pnls = []

        for trade in ticker_trades:
            side = trade.get("side", "").lower()
            qty = int(trade.get("qty", 0))
            price = float(trade.get("price", 0))

            if side == "buy":
                positions.append((qty, price))
            elif side == "sell":
                sell_qty = qty
                sell_price = price
                realized = 0

                while sell_qty > 0 and positions:
                    buy_qty, buy_price = positions[0]
                    match = min(sell_qty, buy_qty)
                    realized += (sell_price - buy_price) * match

                    if buy_qty <= match:
                        positions.pop(0)
                    else:
                        positions[0] = (buy_qty - match, buy_price)

                    sell_qty -= match

                pnls.append(realized)

        return pnls

    @classmethod
    def _fixed_sizing(cls, capital: float, atr: float, mode: str = "CONSERVATIVE") -> Tuple[int, str]:
        """Fallback fixed percentage sizing when Kelly not available."""
        risk_map = {
            "CONSERVATIVE": 0.01,   # 1%
            "BALANCED": 0.02,      # 1.5%
            "AGGRESSIVE": 0.04      # 2%
        }
        risk_fraction = risk_map.get(mode, 0.01)

        rupee_risk = capital * risk_fraction
        risk_per_share = atr * cls.ATR_MULTIPLIER

        if risk_per_share <= 0:
            return 0, "Invalid risk per share"

        qty = int(rupee_risk // risk_per_share)
        return qty, f"Fixed {risk_fraction*100:.1f}% sizing (fallback)"

    @classmethod
    def get_kelly_stats(cls, ticker: str, portfolio: Dict) -> Dict:
        """Return Kelly statistics for display/debugging."""
        pnls = cls._get_ticker_pnls(ticker, portfolio)

        if len(pnls) < cls.MIN_TRADES_FOR_KELLY:
            return {"status": "insufficient_data", "trades": len(pnls)}

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls)
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        payoff = avg_win / avg_loss if avg_loss > 0 else 0

        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0

        return {
            "status": "ok",
            "trades": len(pnls),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff,
            "kelly_fraction": kelly,
            "quarter_kelly": kelly * cls.KELLY_FRACTION
        }


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRATION: Replace fixed sizing in daily_screener.py
# ═════════════════════════════════════════════════════════════════════════════

def integrated_position_sizing_example():
    """
    Example of how to integrate Kelly sizing into your existing screener.

    In your daily_screener.py, replace:

        target_qty = int(rupee_risk_allowed // risk_per_share)

    With:

        from daily_screener import KellyPositionSizer
        target_qty, sizing_reason = KellyPositionSizer.calculate(
            ticker=ticker,
            portfolio=portfolio,
            optimal_params=optimal_params,
            atr=atr,
            current_price=latest_price,
            mode=mode
        )
        print(f"  Position: {sizing_reason}")
    """
    pass


if __name__ == "__main__":
    print("KellyPositionSizer loaded successfully")
    print(f"Kelly fraction: {KellyPositionSizer.KELLY_FRACTION}x")
    print(f"Max risk: {KellyPositionSizer.MAX_RISK_PER_TRADE*100}%")
    print(f"Min trades for Kelly: {KellyPositionSizer.MIN_TRADES_FOR_KELLY}")