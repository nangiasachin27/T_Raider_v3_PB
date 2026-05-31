"""
market_hours/
─────────────
Multi-market trading calendar and hours enforcement.

Quick imports:
    from market_hours import MarketCalendar, guard_or_exit, require_market_open
    from market_hours import MarketOpenContext, OrderGuard, MarketClosedError
    from market_hours import MARKET_CONFIGS, MarketConfig
"""

from market_hours.market_calendar import (
    MarketCalendar,
    MarketConfig,
    TradingStatus,
    MARKET_CONFIGS,
)
from market_hours.trading_guard import (
    guard_or_exit,
    require_market_open,
    MarketOpenContext,
    OrderGuard,
    MarketClosedError,
)

__all__ = [
    "MarketCalendar",
    "MarketConfig",
    "TradingStatus",
    "MARKET_CONFIGS",
    "guard_or_exit",
    "require_market_open",
    "MarketOpenContext",
    "OrderGuard",
    "MarketClosedError",
]