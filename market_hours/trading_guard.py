"""
market_hours/trading_guard.py
────────────────────────────────────────────────────────────────────────────
Trading guard: enforces market hours before any order is placed.

Three usage patterns:

  1. Function decorator — wraps any order-placing function:

        from market_hours.trading_guard import require_market_open

        @require_market_open("INDIA")
        def place_orders():
            ...

  2. Context manager — wraps a block of code:

        from market_hours.trading_guard import MarketOpenContext

        with MarketOpenContext("INDIA"):
            bot.run_autopilot_cycle()

  3. Direct guard call — manual check at any point:

        from market_hours.trading_guard import guard_or_exit

        guard_or_exit("INDIA")   # exits with code 0 if closed

All three raise MarketClosedError when market is closed and
exit_on_closed=False, or call sys.exit(0) when exit_on_closed=True
(the default, so GitHub Actions does not log a failure).
"""

from __future__ import annotations

import sys
import functools
from typing import Optional, Callable, Any
from datetime import datetime

from market_hours.market_calendar import MarketCalendar, TradingStatus, MARKET_CONFIGS


# ════════════════════════════════════════════════════════════════════════════
# EXCEPTION
# ════════════════════════════════════════════════════════════════════════════

class MarketClosedError(Exception):
    """
    Raised when an order is attempted outside market hours.
    Only raised when exit_on_closed=False; otherwise sys.exit(0) is called.
    """
    def __init__(self, status: TradingStatus):
        self.status = status
        super().__init__(str(status))


# ════════════════════════════════════════════════════════════════════════════
# CORE GUARD FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def guard_or_exit(
    market: str = "INDIA",
    exit_on_closed: bool = True,
    at: Optional[datetime] = None,
) -> TradingStatus:
    """
    Check if the market is open. If closed:
      - exit_on_closed=True  → print reason and sys.exit(0)
      - exit_on_closed=False → raise MarketClosedError

    Args:
        market         : Market key from MARKET_CONFIGS e.g. "INDIA", "US"
        exit_on_closed : Whether to exit process (True) or raise (False)
        at             : Datetime to check (default: now)

    Returns:
        TradingStatus if market is open.

    Usage:
        guard_or_exit("INDIA")           # exits if closed
        status = guard_or_exit("US", exit_on_closed=False)  # raises if closed
    """
    cal    = MarketCalendar(market)
    status = cal.get_status(at=at)

    if not status.can_trade:
        print(f"\n{'='*60}")
        print(f"⏸️  TRADING HALTED — {status.market}")
        print(f"{'='*60}")
        print(str(status))
        if status.next_open:
            print(f"  Resuming: {status.next_open.strftime('%A %d %b %Y %H:%M %Z')}")
        print(f"{'='*60}\n")

        if exit_on_closed:
            sys.exit(0)
        else:
            raise MarketClosedError(status)

    print(f"  ✅ {status.reason}")
    return status


# ════════════════════════════════════════════════════════════════════════════
# DECORATOR
# ════════════════════════════════════════════════════════════════════════════

def require_market_open(
    market: str = "INDIA",
    exit_on_closed: bool = True,
) -> Callable:
    """
    Decorator that checks market hours before calling the wrapped function.
    If market is closed, either exits (exit_on_closed=True) or raises
    MarketClosedError (exit_on_closed=False).

    Usage:
        @require_market_open("INDIA")
        def run_autopilot_cycle(mode="CONSERVATIVE"):
            ...

        @require_market_open("US", exit_on_closed=False)
        def place_us_orders():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            guard_or_exit(market=market, exit_on_closed=exit_on_closed)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGER
# ════════════════════════════════════════════════════════════════════════════

class MarketOpenContext:
    """
    Context manager that asserts market is open on entry.

    Usage:
        with MarketOpenContext("INDIA"):
            quarterly_manager.run()

        # With raise instead of exit:
        with MarketOpenContext("US", exit_on_closed=False):
            place_orders()
    """

    def __init__(self, market: str = "INDIA", exit_on_closed: bool = True):
        self.market         = market
        self.exit_on_closed = exit_on_closed
        self.status: Optional[TradingStatus] = None

    def __enter__(self) -> TradingStatus:
        self.status = guard_or_exit(
            market=self.market,
            exit_on_closed=self.exit_on_closed,
        )
        return self.status

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Propagate exceptions — this guard only blocks entry, not exit
        return False


# ════════════════════════════════════════════════════════════════════════════
# ORDER-LEVEL GUARD — used inside execution adapters
# ════════════════════════════════════════════════════════════════════════════

class OrderGuard:
    """
    Lightweight per-order guard used directly inside execution adapters.
    Cached per session — the calendar is only queried once per adapter
    instance regardless of how many orders are placed.

    Usage in upstox_adapter.py / paper_adapter.py:

        class UpstoxExecutionAdapter(ExecutionAdapter):
            def __init__(self, ...):
                ...
                self._order_guard = OrderGuard("INDIA")

            def place_market_order(self, ticker, qty, side, ...):
                self._order_guard.check_or_raise(ticker, side)
                # ... rest of order logic
    """

    def __init__(self, market: str = "INDIA", exit_on_closed: bool = False):
        """
        Args:
            market         : Market key
            exit_on_closed : False = raise MarketClosedError (default for adapters)
                             True  = sys.exit(0) (use at top-level scripts)
        """
        self.market         = market
        self.exit_on_closed = exit_on_closed
        self._cal           = MarketCalendar(market)
        self._cached_status: Optional[TradingStatus] = None
        self._cached_date:   Optional[str]           = None

    def _get_status(self) -> TradingStatus:
        """Return cached status if same calendar day, else re-fetch."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._cached_date != today or self._cached_status is None:
            self._cached_status = self._cal.get_status()
            self._cached_date   = today
        return self._cached_status

    def check_or_raise(self, ticker: str = "", side: str = "") -> TradingStatus:
        """
        Check market is open before placing an order.
        Raises MarketClosedError if closed and exit_on_closed=False.
        """
        status = self._get_status()
        if not status.can_trade:
            action = f"{side} {ticker}".strip() or "order"
            print(
                f"  🚫 ORDER BLOCKED ({action}): {status.reason}"
            )
            if self.exit_on_closed:
                sys.exit(0)
            raise MarketClosedError(status)
        return status

    def is_open(self) -> bool:
        return self._get_status().can_trade