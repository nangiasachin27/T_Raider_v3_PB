"""
execution/adapters/base.py
───────────────────────────
Abstract execution adapter.
All broker implementations must inherit from this.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class OrderResult:
    success: bool
    order_id: str
    status: str  # COMPLETE, REJECTED, PENDING, OPEN, CANCELLED
    filled_qty: int
    avg_price: float
    message: str
    raw_response: dict = None


@dataclass
class PortfolioSnapshot:
    cash: float
    market_value: float
    total_value: float
    holdings: Dict  # {ticker: {"qty": int, "avg_price": float}}


class ExecutionAdapter(ABC):
    """
    Abstract broker interface.
    Implement this for any broker (Upstox, Zerodha, Angel, etc.).
    """
    
    @abstractmethod
    def place_market_order(self, ticker: str, qty: int, side: str, tag: str = "T_Raider") -> OrderResult:
        """
        Place a market order.
        side: "BUY" or "SELL"
        """
        pass
    
    @abstractmethod
    def get_portfolio_snapshot(self, internal_holdings: Dict = None) -> PortfolioSnapshot:
        """
        Fetch real broker holdings + funds.
        internal_holdings: our internal portfolio.json holdings (for reconciliation)
        """
        pass
    
    @abstractmethod
    def get_ltp(self, ticker: str) -> float:
        """Get last traded price for a ticker."""
        pass
    
    @abstractmethod
    def is_order_complete(self, order_id: str) -> bool:
        """Check if order is fully filled."""
        pass