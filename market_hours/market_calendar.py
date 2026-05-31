"""
market_hours/market_calendar.py
────────────────────────────────────────────────────────────────────────────
Multi-market trading calendar and hours guard.

Answers two questions for any supported market:
  1. Is today a trading day (not a holiday, not a weekend)?
  2. Is the current time within the market's trading session?

Built on `exchange_calendars` (pip install exchange-calendars) which uses
the same authoritative holiday data as institutional trading systems.
All holiday lists are maintained upstream — no manual annual updates needed.

Supported markets (extend MARKET_CONFIGS to add more):
  INDIA   — NSE / BSE   (XBOM)   09:15–15:30 IST
  US      — NYSE/NASDAQ (XNYS)   09:30–16:00 ET
  AUS     — ASX         (XASX)   10:00–16:00 AEDT/AEST
  CAN     — TSX         (XTSE)   09:30–16:00 ET

Usage:
    from market_hours.market_calendar import MarketCalendar, TradingStatus

    cal = MarketCalendar("INDIA")
    status = cal.get_status()

    if not status.can_trade:
        print(status.reason)   # "Market closed — Diwali holiday"
        sys.exit(0)

    # Safe to trade
    place_orders(...)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

try:
    import exchange_calendars as ec
except ImportError:
    raise ImportError(
        "exchange_calendars is required. "
        "Install it with: pip install exchange-calendars"
    )


# ════════════════════════════════════════════════════════════════════════════
# MARKET CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketConfig:
    """
    Configuration for a single market.

    Fields:
        name            : Human-readable name e.g. "NSE India"
        calendar_key    : exchange_calendars key e.g. "XBOM"
                          Full list: exchange_calendars.calendar_utils.get_calendar_names()
        timezone        : IANA timezone string e.g. "Asia/Kolkata"
        open_time       : Session open as "HH:MM" in local market time
        close_time      : Session close as "HH:MM" in local market time
        pre_open_buffer : Minutes before open where orders are allowed
                          (useful for AMO — After Market Orders)
        post_close_buffer: Minutes after close where orders are still processed
        currency        : ISO currency code
        lot_size        : Minimum lot size (1 for equities)
    """
    name:               str
    calendar_key:       str
    timezone:           str
    open_time:          str   # "HH:MM"
    close_time:         str   # "HH:MM"
    pre_open_buffer:    int = 0    # minutes
    post_close_buffer:  int = 0    # minutes
    currency:           str = "USD"
    lot_size:           int = 1


# ── Built-in market configs ───────────────────────────────────────────────
# Add any new market here — no other file needs to change.

MARKET_CONFIGS: dict[str, MarketConfig] = {

    "INDIA": MarketConfig(
        name             = "NSE India",
        calendar_key     = "XBOM",        # Bombay Stock Exchange calendar
        timezone         = "Asia/Kolkata",
        open_time        = "09:15",
        close_time       = "15:30",
        pre_open_buffer  = 15,            # 09:00 pre-open session
        post_close_buffer= 0,
        currency         = "INR",
        lot_size         = 1,
    ),

    "US": MarketConfig(
        name             = "NYSE / NASDAQ",
        calendar_key     = "XNYS",
        timezone         = "America/New_York",
        open_time        = "09:30",
        close_time       = "16:00",
        pre_open_buffer  = 0,
        post_close_buffer= 0,
        currency         = "USD",
        lot_size         = 1,
    ),

    "AUS": MarketConfig(
        name             = "ASX Australia",
        calendar_key     = "XASX",
        timezone         = "Australia/Sydney",
        open_time        = "10:00",
        close_time       = "16:00",
        pre_open_buffer  = 0,
        post_close_buffer= 0,
        currency         = "AUD",
        lot_size         = 1,
    ),

    "CAN": MarketConfig(
        name             = "TSX Canada",
        calendar_key     = "XTSE",
        timezone         = "America/Toronto",
        open_time        = "09:30",
        close_time       = "16:00",
        pre_open_buffer  = 0,
        post_close_buffer= 0,
        currency         = "CAD",
        lot_size         = 1,
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TradingStatus:
    """
    Result of a market status check.

    Attributes:
        can_trade       : True if orders can be placed right now
        market          : Market identifier e.g. "INDIA"
        local_time      : Current time in market's local timezone
        is_trading_day  : True if today is not a holiday/weekend
        is_open         : True if within session hours (incl. buffers)
        reason          : Human-readable explanation (always populated)
        next_open       : When the market next opens (if currently closed)
        holiday_name    : Name of today's holiday if applicable
    """
    can_trade:      bool
    market:         str
    local_time:     datetime
    is_trading_day: bool
    is_open:        bool
    reason:         str
    next_open:      Optional[datetime] = None
    holiday_name:   Optional[str]      = None

    def __str__(self):
        status = "✅ OPEN" if self.can_trade else "🚫 CLOSED"
        parts = [
            f"{status} — {self.market} ({self.local_time.strftime('%Y-%m-%d %H:%M %Z')})",
            f"  {self.reason}",
        ]
        if self.next_open:
            parts.append(f"  Next open: {self.next_open.strftime('%Y-%m-%d %H:%M %Z')}")
        return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# MARKET CALENDAR
# ════════════════════════════════════════════════════════════════════════════

class MarketCalendar:
    """
    Trading calendar and hours guard for a single market.

    Example:
        cal = MarketCalendar("INDIA")
        status = cal.get_status()
        if not status.can_trade:
            print(status)
            sys.exit(0)
    """

    def __init__(self, market: str = "INDIA"):
        if market not in MARKET_CONFIGS:
            available = list(MARKET_CONFIGS.keys())
            raise ValueError(
                f"Unknown market '{market}'. "
                f"Available: {available}. "
                f"Add a new MarketConfig to MARKET_CONFIGS to extend."
            )
        self.market = market
        self.config = MARKET_CONFIGS[market]
        self.tz     = ZoneInfo(self.config.timezone)

        # Load exchange_calendars calendar (cached after first load)
        self._calendar = ec.get_calendar(self.config.calendar_key)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _now_local(self) -> datetime:
        """Current datetime in market's local timezone."""
        return datetime.now(tz=self.tz)

    def _parse_time(self, hhmm: str, ref_date: date) -> datetime:
        """Parse "HH:MM" into a timezone-aware datetime on ref_date."""
        h, m = map(int, hhmm.split(":"))
        return datetime(ref_date.year, ref_date.month, ref_date.day,
                        h, m, tzinfo=self.tz)

    def _is_session_day(self, check_date: date) -> tuple[bool, Optional[str]]:
        """
        Returns (is_trading_day, holiday_name_or_None).
        Uses exchange_calendars authoritative schedule.
        """
        try:
            import pandas as pd
            pd_date = pd.Timestamp(check_date)
            is_session = self._calendar.is_session(pd_date)
            if is_session:
                return True, None
            # Not a session — check if it's a named holiday
            holiday_name = self._get_holiday_name(check_date)
            return False, holiday_name
        except Exception:
            # Fallback: basic weekend check if exchange_calendars fails
            if check_date.weekday() >= 5:
                return False, "Weekend"
            return True, None

    def _get_holiday_name(self, check_date: date) -> Optional[str]:
        """Try to get the name of the holiday on check_date."""
        try:
            import pandas as pd
            # exchange_calendars exposes holiday names via adhoc_holidays
            # and regular_holidays depending on calendar implementation
            cal = self._calendar
            ts = pd.Timestamp(check_date)

            # Try to find named holiday
            if hasattr(cal, 'adhoc_holidays'):
                for h in cal.adhoc_holidays:
                    if pd.Timestamp(h).date() == check_date:
                        return "Public holiday"

            # Weekend check
            if check_date.weekday() == 5:
                return "Saturday"
            if check_date.weekday() == 6:
                return "Sunday"

            return "Market holiday"
        except Exception:
            return "Market holiday"

    def _next_open(self) -> Optional[datetime]:
        """Return the next session open datetime in local time."""
        try:
            import pandas as pd
            today = self._now_local().date()
            search_start = pd.Timestamp(today)
            # Look up to 10 calendar days ahead
            search_end   = pd.Timestamp(today + timedelta(days=10))
            sessions = self._calendar.sessions_in_range(search_start, search_end)

            for sess in sessions:
                sess_date = sess.date()
                if sess_date >= today:
                    open_utc = self._calendar.session_open(sess)
                    return open_utc.astimezone(self.tz)
            return None
        except Exception:
            return None

    # ── Public API ────────────────────────────────────────────────────────

    def get_status(self, at: Optional[datetime] = None) -> TradingStatus:
        """
        Check whether trading is currently allowed.

        Args:
            at: datetime to check (default: now). Must be timezone-aware if provided.

        Returns:
            TradingStatus with can_trade, reason, and next_open.
        """
        now = at or self._now_local()
        if now.tzinfo is None:
            now = now.replace(tzinfo=self.tz)
        now_local = now.astimezone(self.tz)
        today     = now_local.date()

        # ── Check 1: Is today a trading day? ─────────────────────────────
        is_trading_day, holiday_name = self._is_session_day(today)

        if not is_trading_day:
            hname = holiday_name or "Market holiday"
            return TradingStatus(
                can_trade      = False,
                market         = self.market,
                local_time     = now_local,
                is_trading_day = False,
                is_open        = False,
                reason         = f"Market closed — {hname}",
                next_open      = self._next_open(),
                holiday_name   = holiday_name,
            )

        # ── Check 2: Is current time within session hours? ────────────────
        session_open  = self._parse_time(self.config.open_time,  today)
        session_close = self._parse_time(self.config.close_time, today)

        # Apply buffers
        effective_open  = session_open  - timedelta(minutes=self.config.pre_open_buffer)
        effective_close = session_close + timedelta(minutes=self.config.post_close_buffer)

        if now_local < effective_open:
            return TradingStatus(
                can_trade      = False,
                market         = self.market,
                local_time     = now_local,
                is_trading_day = True,
                is_open        = False,
                reason         = (
                    f"Market not yet open — opens at "
                    f"{session_open.strftime('%H:%M %Z')} "
                    f"(current: {now_local.strftime('%H:%M')})"
                ),
                next_open      = effective_open,
            )

        if now_local > effective_close:
            return TradingStatus(
                can_trade      = False,
                market         = self.market,
                local_time     = now_local,
                is_trading_day = True,
                is_open        = False,
                reason         = (
                    f"Market closed for the day — closed at "
                    f"{session_close.strftime('%H:%M %Z')} "
                    f"(current: {now_local.strftime('%H:%M')})"
                ),
                next_open      = self._next_open(),
            )

        # ── All checks passed ─────────────────────────────────────────────
        time_to_close = effective_close - now_local
        minutes_left  = int(time_to_close.total_seconds() / 60)

        return TradingStatus(
            can_trade      = True,
            market         = self.market,
            local_time     = now_local,
            is_trading_day = True,
            is_open        = True,
            reason         = (
                f"Market open — "
                f"{self.config.open_time}–{self.config.close_time} "
                f"{self.config.timezone.split('/')[-1]} "
                f"({minutes_left} min remaining)"
            ),
        )

    def is_open(self, at: Optional[datetime] = None) -> bool:
        """Convenience method — returns True if trading is allowed."""
        return self.get_status(at).can_trade

    def assert_open(self, exit_on_closed: bool = True) -> TradingStatus:
        """
        Assert market is open. Prints status and optionally exits.

        Args:
            exit_on_closed: If True, calls sys.exit(0) when market is closed.
                            Exit code 0 so GitHub Actions does not report failure.

        Returns:
            TradingStatus (only returns if market is open, or exit_on_closed=False)
        """
        status = self.get_status()
        print(str(status))
        if not status.can_trade and exit_on_closed:
            sys.exit(0)
        return status