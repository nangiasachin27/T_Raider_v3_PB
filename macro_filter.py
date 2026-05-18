"""
macro_filter.py — Market-Agnostic Macro Filter
═══════════════════════════════════════════════════════════════════════════════
Provides three pre-trade environment checks that sit ABOVE the technical
signal layer in daily_screener.py:

    1. Volatility Gate      — Is fear elevated? (VIX / equivalent)
    2. Institutional Flow   — Are big players buying or selling net?
    3. Earnings Blackout    — Is this stock near a result date?

MARKET AGNOSTIC DESIGN
───────────────────────
All market-specific configuration lives in a MarketConfig dataclass.
To add a new market, define one config block — no logic changes required.

Currently supported:
    INDIA      — NSE/BSE (Nifty 50, India VIX, FII/DII via NSE)
    AUSTRALIA  — ASX (ASX 200, XVI volatility index)
    CANADA     — TSX (S&P/TSX, VIXC volatility index)
    USA        — NYSE/NASDAQ (S&P 500, VIX)

Adding a new market:
    1. Add a MarketConfig entry to MARKET_CONFIGS
    2. Add an institutional flow fetcher to _fetch_institutional_flow()
       (or return None if unavailable — the filter degrades gracefully)
    3. Done. All three gates work automatically.

INTEGRATION
───────────────────────
In daily_screener.py, replace the existing regime block with:

    from macro_filter import MacroFilter, MarketConfig, MARKET_CONFIGS

    macro = MacroFilter(MARKET_CONFIGS["INDIA"])
    macro_context = macro.run()               # fetch once per session
    macro.print_summary(macro_context)

Then inside the BUY gate (after Gate 1: Regime), add Gate 1b:

    macro_result = macro.evaluate_buy(ticker, macro_context)
    if macro_result.action == "SKIP":
        all_signals.append(ScreenerSignal(..., reason=macro_result.reason))
        continue
    if macro_result.action == "DOWNGRADE":
        # confidence_tier will be capped at MEDIUM downstream
        pass   # macro_result.reason is appended to the signal reason

GRACEFUL DEGRADATION
───────────────────────
Every fetch is wrapped in try/except. If a data source is unavailable
(weekend, API issue, new market), that filter returns a NEUTRAL result
and the screener continues normally. Failures are logged, never silent.

DEPENDENCIES
───────────────────────
    yfinance    — VIX + earnings calendar (all markets)
    requests    — NSE institutional flow page (India only)
    pandas      — standard
    All already in requirements.txt
"""

from __future__ import annotations

import datetime
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
import yfinance as yf


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Market Configuration — add new markets here only
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MarketConfig:
    """
    Everything that varies between markets. Add a new entry to MARKET_CONFIGS
    to support a new exchange — no changes to filter logic required.

    Fields
    ──────
    name                : Human-readable label e.g. "India (NSE)"
    vix_ticker          : yfinance symbol for the volatility index
    vix_elevated        : VIX level above which fear is "elevated" → DOWNGRADE
    vix_extreme         : VIX level above which fear is "extreme"  → SKIP
    benchmark_ticker    : yfinance symbol for the market index (for overnight)
    overnight_ref_ticker: yfinance symbol for the US market used as overnight
                          reference. Set to None for US markets themselves.
    flow_source         : Institutional flow data source identifier.
                          "NSE_WEB"  = India FII/DII (scraped from NSE)
                          "NONE"     = Not available for this market
    flow_bearish_threshold: Net institutional flow (in millions, local currency)
                          below which environment is bearish. Negative = net sell.
    earnings_buffer_days: Trading days before earnings where buys are downgraded.
    timezone            : Market timezone string (used for calendar day checks)
    high_risk_dates     : List of "YYYY-MM-DD" strings for known event days
                          (central bank meetings, budget etc.)
                          Tip: update this list at the start of each year.
    currency_symbol     : For display formatting only.
    """
    name:                   str
    vix_ticker:             str
    vix_elevated:           float
    vix_extreme:            float
    benchmark_ticker:       str
    overnight_ref_ticker:   Optional[str]
    flow_source:            str
    flow_bearish_threshold: float
    earnings_buffer_days:   int
    timezone:               str
    high_risk_dates:        tuple
    currency_symbol:        str


MARKET_CONFIGS: dict[str, MarketConfig] = {

    "INDIA": MarketConfig(
        name                   = "India (NSE/BSE)",
        vix_ticker             = "^INDIAVIX",
        vix_elevated           = 16.0,
        vix_extreme            = 22.0,
        benchmark_ticker       = "^NSEI",
        overnight_ref_ticker   = "^GSPC",       # S&P 500
        flow_source            = "NSE_WEB",
        flow_bearish_threshold = -2000.0,        # ₹ crore
        earnings_buffer_days   = 5,
        timezone               = "Asia/Kolkata",
        high_risk_dates        = (
            # RBI MPC 2026 expected dates (update annually)
            "2026-04-09", "2026-06-06", "2026-08-08",
            "2026-10-08", "2026-12-05",
            # Union Budget
            "2027-02-01",
        ),
        currency_symbol        = "₹",
    ),

    "AUSTRALIA": MarketConfig(
        name                   = "Australia (ASX)",
        vix_ticker             = "^XVI",         # ASX Volatility Index
        vix_elevated           = 16.0,
        vix_extreme            = 22.0,
        benchmark_ticker       = "^AXJO",        # ASX 200
        overnight_ref_ticker   = "^GSPC",        # S&P 500
        flow_source            = "NONE",         # No public equivalent of FII/DII
        flow_bearish_threshold = 0.0,
        earnings_buffer_days   = 3,
        timezone               = "Australia/Sydney",
        high_risk_dates        = (
            # RBA board meeting dates 2026 (update annually)
            "2026-02-03", "2026-03-31", "2026-05-19",
            "2026-07-07", "2026-08-04", "2026-09-01",
            "2026-10-06", "2026-11-03", "2026-12-01",
            # Federal Budget
            "2026-05-12",
        ),
        currency_symbol        = "A$",
    ),

    "CANADA": MarketConfig(
        name                   = "Canada (TSX)",
        vix_ticker             = "^VIXC",        # S&P/TSX 60 VIX
        vix_elevated           = 18.0,
        vix_extreme            = 25.0,
        benchmark_ticker       = "^GSPTSE",      # S&P/TSX Composite
        overnight_ref_ticker   = "^GSPC",        # S&P 500
        flow_source            = "NONE",
        flow_bearish_threshold = 0.0,
        earnings_buffer_days   = 3,
        timezone               = "America/Toronto",
        high_risk_dates        = (
            # Bank of Canada rate decisions 2026 (update annually)
            "2026-01-29", "2026-03-04", "2026-04-15",
            "2026-06-03", "2026-07-15", "2026-09-09",
            "2026-10-28", "2026-12-09",
            # Federal Budget
            "2026-03-25",
        ),
        currency_symbol        = "C$",
    ),

    "USA": MarketConfig(
        name                   = "USA (NYSE/NASDAQ)",
        vix_ticker             = "^VIX",
        vix_elevated           = 20.0,
        vix_extreme            = 30.0,
        benchmark_ticker       = "^GSPC",
        overnight_ref_ticker   = None,           # IS the reference market
        flow_source            = "NONE",
        flow_bearish_threshold = 0.0,
        earnings_buffer_days   = 3,
        timezone               = "America/New_York",
        high_risk_dates        = (
            # FOMC meeting dates 2026 (update annually)
            "2026-01-29", "2026-03-19", "2026-05-07",
            "2026-06-18", "2026-07-30", "2026-09-17",
            "2026-11-05", "2026-12-16",
        ),
        currency_symbol        = "$",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Result types
# ═══════════════════════════════════════════════════════════════════════════════

class FilterAction(str, Enum):
    PASS      = "PASS"       # Signal proceeds normally
    DOWNGRADE = "DOWNGRADE"  # Signal continues but confidence capped at MEDIUM
    SKIP      = "SKIP"       # Signal blocked entirely


@dataclass
class FilterResult:
    """Outcome of a single filter check."""
    name:   str
    action: FilterAction
    reason: str
    value:  Optional[float] = None    # raw metric value for logging


@dataclass
class MacroContext:
    """
    Snapshot of all macro data fetched once per screener session.
    Passed into evaluate_buy() for each ticker — no repeated network calls.
    """
    vix:                  Optional[float]        = None
    vix_status:           str                    = "UNKNOWN"
    institutional_flow:   Optional[float]        = None
    flow_status:          str                    = "UNAVAILABLE"
    overnight_change_pct: Optional[float]        = None
    is_high_risk_day:     bool                   = False
    high_risk_reason:     str                    = ""
    fetch_errors:         list                   = field(default_factory=list)
    fetched_at:           str                    = ""

    # Per-ticker earnings data is fetched lazily inside evaluate_buy()
    # and cached here to avoid repeated yfinance calls in the same session.
    _earnings_cache:      dict                   = field(default_factory=dict)


@dataclass
class BuyEvaluation:
    """
    Aggregated result for a specific ticker at a specific moment.
    Returned by MacroFilter.evaluate_buy().
    """
    ticker:           str
    action:           FilterAction
    reason:           str
    filter_results:   list[FilterResult] = field(default_factory=list)

    def warning_flags(self) -> list[str]:
        return [r.reason for r in self.filter_results if r.action != FilterAction.PASS]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MacroFilter — the main class
# ═══════════════════════════════════════════════════════════════════════════════

class MacroFilter:
    """
    Market-agnostic macro environment filter.

    Usage
    ─────
        from macro_filter import MacroFilter, MARKET_CONFIGS

        macro   = MacroFilter(MARKET_CONFIGS["INDIA"])
        context = macro.run()           # fetch all data once per session
        macro.print_summary(context)

        # Inside the per-ticker BUY gate:
        evaluation = macro.evaluate_buy("RELIANCE.NS", context)
        if evaluation.action == FilterAction.SKIP:
            # block the signal
        elif evaluation.action == FilterAction.DOWNGRADE:
            # cap confidence tier at MEDIUM
    """

    def __init__(self, config: MarketConfig):
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> MacroContext:
        """
        Fetch all session-level macro data. Call once at the start of
        daily_screener.run_screener(), not once per ticker.
        """
        ctx = MacroContext(fetched_at=datetime.datetime.now().isoformat())

        ctx.vix, ctx.vix_status             = self._fetch_vix()
        ctx.institutional_flow, ctx.flow_status = self._fetch_institutional_flow()
        ctx.overnight_change_pct            = self._fetch_overnight_change()
        ctx.is_high_risk_day, ctx.high_risk_reason = self._check_calendar()

        return ctx

    def evaluate_buy(self, ticker: str, ctx: MacroContext) -> BuyEvaluation:
        """
        Evaluate whether a BUY signal for `ticker` should proceed given
        the current macro context. Returns PASS, DOWNGRADE, or SKIP with
        a human-readable reason.

        Hierarchy (worst outcome wins):
            Any SKIP    → overall SKIP
            Any DOWNGRADE (no SKIP) → overall DOWNGRADE
            All PASS    → overall PASS
        """
        results: list[FilterResult] = []

        results.append(self._gate_vix(ctx))
        results.append(self._gate_institutional_flow(ctx))
        results.append(self._gate_earnings(ticker, ctx))
        results.append(self._gate_calendar(ctx))
        results.append(self._gate_overnight(ctx))

        # Worst outcome wins
        if any(r.action == FilterAction.SKIP for r in results):
            worst    = next(r for r in results if r.action == FilterAction.SKIP)
            action   = FilterAction.SKIP
            reason   = worst.reason
        elif any(r.action == FilterAction.DOWNGRADE for r in results):
            flags    = [r.reason for r in results if r.action == FilterAction.DOWNGRADE]
            action   = FilterAction.DOWNGRADE
            reason   = " | ".join(flags)
        else:
            action   = FilterAction.PASS
            reason   = "All macro gates passed"

        return BuyEvaluation(
            ticker=ticker, action=action,
            reason=reason, filter_results=results,
        )

    def print_summary(self, ctx: MacroContext) -> None:
        """Print a clean macro environment summary for the screener output."""
        c = self.config
        print(f"\n{'─'*55}")
        print(f"🌍 MACRO ENVIRONMENT — {c.name}")
        print(f"{'─'*55}")

        # VIX
        if ctx.vix is not None:
            emoji = "🟢" if ctx.vix_status == "NORMAL" else ("🟡" if ctx.vix_status == "ELEVATED" else "🔴")
            print(f"  Volatility (VIX)  : {emoji} {ctx.vix:.1f}  [{ctx.vix_status}]")
        else:
            print(f"  Volatility (VIX)  : ⚪ UNAVAILABLE")

        # Institutional flow
        if ctx.institutional_flow is not None:
            flow_emoji = "🟢" if ctx.institutional_flow >= 0 else "🔴"
            print(f"  Institutional Flow: {flow_emoji} {c.currency_symbol}{ctx.institutional_flow:+,.0f}M  [{ctx.flow_status}]")
        else:
            print(f"  Institutional Flow: ⚪ {ctx.flow_status}")

        # Overnight
        if ctx.overnight_change_pct is not None:
            ov_emoji = "🟢" if ctx.overnight_change_pct >= -1.0 else ("🟡" if ctx.overnight_change_pct >= -1.5 else "🔴")
            print(f"  Overnight (Ref)   : {ov_emoji} {ctx.overnight_change_pct:+.2f}%")
        else:
            print(f"  Overnight (Ref)   : ⚪ UNAVAILABLE")

        # Calendar
        if ctx.is_high_risk_day:
            print(f"  Calendar          : ⚠️  HIGH-RISK DAY — {ctx.high_risk_reason}")
        else:
            print(f"  Calendar          : 🟢 No scheduled risk events today")

        if ctx.fetch_errors:
            print(f"  ⚠️  Fetch warnings : {', '.join(ctx.fetch_errors)}")

        print(f"{'─'*55}\n")

    # ── Filter gates ──────────────────────────────────────────────────────────

    def _gate_vix(self, ctx: MacroContext) -> FilterResult:
        """
        FILTER 1 — Volatility Gate
        ──────────────────────────
        High volatility = wider spreads, larger gaps, stop-hunts.
        Even a technically valid signal is harder to execute profitably
        when fear is elevated.

        NORMAL   (VIX < elevated)  : PASS
        ELEVATED (VIX >= elevated) : DOWNGRADE — signal proceeds, confidence capped
        EXTREME  (VIX >= extreme)  : SKIP — new positions too risky

        Rationale for thresholds (configurable per market):
            India VIX: normal range 12-16; elevated 16-22; extreme 22+
            US VIX   : normal range 12-20; elevated 20-30; extreme 30+
        """
        if ctx.vix is None:
            return FilterResult("VIX", FilterAction.PASS,
                                "VIX unavailable — filter skipped (safe default)")

        c = self.config
        if ctx.vix >= c.vix_extreme:
            return FilterResult(
                "VIX", FilterAction.SKIP,
                f"⛔ VIX EXTREME ({ctx.vix:.1f} ≥ {c.vix_extreme}) — "
                f"market fear too high for new positions",
                value=ctx.vix,
            )
        if ctx.vix >= c.vix_elevated:
            return FilterResult(
                "VIX", FilterAction.DOWNGRADE,
                f"⚠️ VIX ELEVATED ({ctx.vix:.1f} ≥ {c.vix_elevated}) — "
                f"confidence capped at MEDIUM",
                value=ctx.vix,
            )
        return FilterResult("VIX", FilterAction.PASS,
                            f"VIX normal ({ctx.vix:.1f})", value=ctx.vix)

    def _gate_institutional_flow(self, ctx: MacroContext) -> FilterResult:
        """
        FILTER 2 — Institutional Flow Gate (India: FII/DII net flow)
        ─────────────────────────────────────────────────────────────
        Retail signals against institutional money flow rarely work out.
        FIIs (Foreign Institutional Investors) drive Nifty direction.

        POSITIVE or MILD NEGATIVE flow : PASS
        BEARISH (flow < threshold)      : DOWNGRADE

        We don't SKIP on flow alone because DII buying can offset FII
        selling — net flow is the right metric, not either in isolation.
        We only SKIP on VIX (fear is concrete) or calendar (known event).

        For markets where flow data is unavailable, this gate always PASSes.
        """
        if ctx.institutional_flow is None:
            return FilterResult("FLOW", FilterAction.PASS,
                                f"Institutional flow: {ctx.flow_status}")

        threshold = self.config.flow_bearish_threshold
        if threshold == 0.0:
            # Market doesn't define a threshold — no-op
            return FilterResult("FLOW", FilterAction.PASS, "Flow gate not configured for this market")

        if ctx.institutional_flow < threshold:
            return FilterResult(
                "FLOW", FilterAction.DOWNGRADE,
                f"⚠️ Bearish institutional flow "
                f"({self.config.currency_symbol}{ctx.institutional_flow:+,.0f}M < "
                f"threshold {self.config.currency_symbol}{threshold:,.0f}M) — "
                f"confidence capped at MEDIUM",
                value=ctx.institutional_flow,
            )
        return FilterResult(
            "FLOW", FilterAction.PASS,
            f"Institutional flow net positive/neutral "
            f"({self.config.currency_symbol}{ctx.institutional_flow:+,.0f}M)",
            value=ctx.institutional_flow,
        )

    def _gate_earnings(self, ticker: str, ctx: MacroContext) -> FilterResult:
        """
        FILTER 3 — Earnings Blackout Gate
        ──────────────────────────────────
        Buying into a stock 1-5 days before its quarterly results is a
        coin flip regardless of how strong the technical setup looks.
        The technical signal reflects pre-earnings positioning, not your
        strategy working. Post-result gaps routinely exceed stop-loss levels.

        Within earnings_buffer_days of result : DOWNGRADE
        On earnings day itself                : SKIP

        Result is cached in ctx._earnings_cache to avoid fetching the same
        ticker twice if the screener evaluates it across multiple strategies.
        """
        buffer = self.config.earnings_buffer_days
        today  = datetime.date.today()

        # Cache check
        if ticker in ctx._earnings_cache:
            days_to_earnings = ctx._earnings_cache[ticker]
        else:
            days_to_earnings = self._days_to_next_earnings(ticker)
            ctx._earnings_cache[ticker] = days_to_earnings

        if days_to_earnings is None:
            return FilterResult("EARNINGS", FilterAction.PASS,
                                "Earnings date unknown — filter skipped")

        if days_to_earnings == 0:
            return FilterResult(
                "EARNINGS", FilterAction.SKIP,
                f"⛔ EARNINGS TODAY — do not open new position",
                value=float(days_to_earnings),
            )
        if days_to_earnings <= buffer:
            return FilterResult(
                "EARNINGS", FilterAction.DOWNGRADE,
                f"⚠️ Earnings in {days_to_earnings} trading day(s) — "
                f"gap risk elevated, confidence capped at MEDIUM",
                value=float(days_to_earnings),
            )
        return FilterResult(
            "EARNINGS", FilterAction.PASS,
            f"Next earnings >{buffer} days away ({days_to_earnings}d)",
            value=float(days_to_earnings),
        )

    def _gate_calendar(self, ctx: MacroContext) -> FilterResult:
        """
        FILTER 4 — High-Risk Calendar Gate
        ────────────────────────────────────
        Central bank meetings (RBI, RBA, Fed, BoC) and budget days produce
        outsized intraday moves. Opening new positions on these days means
        accepting event risk that has nothing to do with your strategy edge.

        Result: DOWNGRADE (not SKIP — experienced traders do trade events,
        but it should be a conscious choice, not an accidental one).
        """
        if ctx.is_high_risk_day:
            return FilterResult(
                "CALENDAR", FilterAction.DOWNGRADE,
                f"⚠️ High-risk calendar event: {ctx.high_risk_reason}",
            )
        return FilterResult("CALENDAR", FilterAction.PASS, "No calendar events today")

    def _gate_overnight(self, ctx: MacroContext) -> FilterResult:
        """
        FILTER 5 — Overnight Reference Market Gate
        ────────────────────────────────────────────
        For markets that open after the US close (India, Australia, Canada),
        a sharp overnight drop in S&P 500 usually leads to gap-downs at open.
        Opening new longs into a guaranteed gap-down is poor execution.

        US markets: this gate is skipped (no overnight reference).

        -0% to -1.5% : PASS   (normal noise)
        -1.5% to -2%  : DOWNGRADE
        Below -2%     : SKIP  (expect significant gap-down at open)
        """
        if self.config.overnight_ref_ticker is None:
            return FilterResult("OVERNIGHT", FilterAction.PASS,
                                "Overnight gate not applicable for this market")

        if ctx.overnight_change_pct is None:
            return FilterResult("OVERNIGHT", FilterAction.PASS,
                                "Overnight data unavailable — filter skipped")

        chg = ctx.overnight_change_pct
        if chg <= -2.0:
            return FilterResult(
                "OVERNIGHT", FilterAction.SKIP,
                f"⛔ Ref market down {chg:.2f}% overnight — "
                f"significant gap-down expected at open",
                value=chg,
            )
        if chg <= -1.5:
            return FilterResult(
                "OVERNIGHT", FilterAction.DOWNGRADE,
                f"⚠️ Ref market down {chg:.2f}% overnight — "
                f"gap-down risk, confidence capped at MEDIUM",
                value=chg,
            )
        return FilterResult(
            "OVERNIGHT", FilterAction.PASS,
            f"Ref market overnight change: {chg:+.2f}%",
            value=chg,
        )

    # ── Data fetchers ─────────────────────────────────────────────────────────

    def _fetch_vix(self) -> tuple[Optional[float], str]:
        """Fetch VIX (or market-equivalent volatility index) via yfinance."""
        try:
            ticker = self.config.vix_ticker
            data   = yf.Ticker(ticker).history(period="2d")
            if data.empty:
                return None, "UNAVAILABLE"
            vix = float(data['Close'].iloc[-1])
            c   = self.config
            status = ("EXTREME"  if vix >= c.vix_extreme  else
                      "ELEVATED" if vix >= c.vix_elevated else
                      "NORMAL")
            return vix, status
        except Exception as e:
            warnings.warn(f"VIX fetch failed ({self.config.vix_ticker}): {e}")
            return None, "FETCH_ERROR"

    def _fetch_institutional_flow(self) -> tuple[Optional[float], str]:
        """
        Fetch net institutional flow for the configured market.
        Dispatches to the appropriate source based on config.flow_source.

        Returns (net_flow_in_millions, status_string).
        net_flow > 0 = net buying, < 0 = net selling.
        """
        source = self.config.flow_source

        if source == "NSE_WEB":
            return self._fetch_nse_fii_dii()
        elif source == "NONE":
            return None, "NOT_AVAILABLE_FOR_THIS_MARKET"
        else:
            return None, f"UNKNOWN_SOURCE:{source}"

    def _fetch_nse_fii_dii(self) -> tuple[Optional[float], str]:
        """
        Fetch India FII + DII net equity flow from NSE.
        Returns combined net flow in ₹ crore (positive = net buying).

        NSE publishes this at: https://www.nseindia.com/api/fiidiiTradeReact
        The page is JSON-first but requires a session cookie from the main
        NSE site — hence the two-step request with a browser-like header.

        Falls back gracefully if NSE blocks the request or changes the format.
        """
        try:
            import requests

            session = requests.Session()
            session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.nseindia.com/",
            })

            # Step 1: Seed the cookie
            session.get("https://www.nseindia.com", timeout=10)

            # Step 2: Fetch FII/DII data
            resp = session.get(
                "https://www.nseindia.com/api/fiidiiTradeReact",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            # ── DEFENSIVE: Handle unexpected response types ─────────────
            # NSE sometimes returns HTML, string error, or malformed JSON
            if isinstance(data, str):
                warnings.warn(f"NSE FII/DII returned string instead of JSON list: {data[:200]}")
                return None, "NSE_STRING_RESPONSE"
            
            if isinstance(data, dict):
                # Sometimes NSE wraps data in a dict with metadata
                # Try common keys, or just return error
                warnings.warn(f"NSE FII/DII returned dict instead of list: {list(data.keys())}")
                return None, "NSE_DICT_RESPONSE"

            if not isinstance(data, list):
                warnings.warn(f"NSE FII/DII returned unexpected type: {type(data)}")
                return None, "NSE_UNKNOWN_FORMAT"

            # NSE returns a list; latest day is first entry
            # Each entry should be a list of row dicts
            today_rows = data[0] if len(data) > 0 else []

            # ── DEFENSIVE: Ensure today_rows is iterable of dicts ───────
            if isinstance(today_rows, str):
                warnings.warn(f"NSE FII/DII first element is string: {today_rows[:200]}")
                return None, "NSE_MALFORMED_LIST"
            
            if isinstance(today_rows, dict):
                # Single row wrapped in dict — normalize to list
                today_rows = [today_rows]
            
            if not isinstance(today_rows, list):
                warnings.warn(f"NSE FII/DII rows have unexpected type: {type(today_rows)}")
                return None, "NSE_ROWS_NOT_LIST"

            if not today_rows:
                return None, "NSE_EMPTY_RESPONSE"

            net_fii = 0.0
            net_dii = 0.0
            for row in today_rows:
                # ── DEFENSIVE: Skip non-dict rows ───────────────────────
                if not isinstance(row, dict):
                    warnings.warn(f"Skipping non-dict row in FII/DII data: {type(row)} = {str(row)[:100]}")
                    continue

                category = row.get("category", "").upper()
                try:
                    net = float(str(row.get("netValue", "0")).replace(",", ""))
                except (ValueError, TypeError):
                    net = 0.0
                if "FII" in category or "FPI" in category:
                    net_fii += net
                elif "DII" in category:
                    net_dii += net

            combined_net = net_fii + net_dii
            status = "BULLISH" if combined_net >= 0 else "BEARISH"
            return combined_net, status

        except Exception as e:
            warnings.warn(f"NSE FII/DII fetch failed: {e}")
            return None, "NSE_FETCH_ERROR"

    def _fetch_overnight_change(self) -> Optional[float]:
        """
        Fetch the most recent daily change of the overnight reference market.
        Returns percentage change, e.g. -1.5 means down 1.5%.
        Returns None on failure (gate will PASS by default).
        """
        ref = self.config.overnight_ref_ticker
        if ref is None:
            return None
        try:
            data = yf.Ticker(ref).history(period="3d")
            if len(data) < 2:
                return None
            prev  = float(data['Close'].iloc[-2])
            last  = float(data['Close'].iloc[-1])
            return (last / prev - 1) * 100 if prev > 0 else None
        except Exception as e:
            warnings.warn(f"Overnight reference fetch failed ({ref}): {e}")
            return None

    def _check_calendar(self) -> tuple[bool, str]:
        """Check if today is a configured high-risk event day."""
        today = datetime.date.today().isoformat()
        if today in self.config.high_risk_dates:
            return True, f"Scheduled policy/event day ({today})"
        return False, ""

    def _days_to_next_earnings(self, ticker: str) -> Optional[int]:
        """
        Returns the number of calendar days until the next earnings date,
        or None if unavailable.

        yfinance .calendar is the most accessible cross-market source for
        this, though coverage varies by exchange. For Indian NSE stocks the
        coverage is patchy — the filter degrades gracefully (returns None →
        PASS) when no data is available.
        """
        try:
            cal = yf.Ticker(ticker).calendar
            if cal is None:
                return None

            # yfinance returns a DataFrame with columns as dates for some tickers
            # and a dict for others depending on version. Handle both.
            earnings_dates = []
            if isinstance(cal, dict) and "Earnings Date" in cal:
                raw = cal["Earnings Date"]
                earnings_dates = raw if isinstance(raw, list) else [raw]
            elif hasattr(cal, "columns"):
                # DataFrame format — earnings date is in the column headers
                for col in cal.columns:
                    try:
                        d = pd.to_datetime(col).date()
                        earnings_dates.append(d)
                    except Exception:
                        continue

            today = datetime.date.today()
            future = [
                d.date() if hasattr(d, "date") else d
                for d in earnings_dates
                if (d.date() if hasattr(d, "date") else d) >= today
            ]
            if not future:
                return None
            nearest = min(future)
            return (nearest - today).days

        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Convenience helper for daily_screener.py integration
# ═══════════════════════════════════════════════════════════════════════════════

def apply_macro_filter_to_signal(
    ticker:         str,
    confidence_tier: str,
    reason:         str,
    macro:          MacroFilter,
    ctx:            MacroContext,
) -> tuple[str, str]:
    """
    Drop-in helper for daily_screener.py BUY gate.

    Given the existing confidence_tier and reason from the technical signal,
    applies the macro evaluation and returns updated (confidence_tier, reason).

    Usage in daily_screener.py classify_buy_confidence block:

        confidence_tier, reason = classify_buy_confidence(...)
        if confidence_tier != "SKIP":
            confidence_tier, reason = apply_macro_filter_to_signal(
                ticker, confidence_tier, reason, macro, macro_context
            )

    Returns:
        ("SKIP",   updated_reason)  if macro says SKIP
        ("MEDIUM", updated_reason)  if macro says DOWNGRADE and tier was HIGH
        (original_tier, updated_reason + macro flags)  if macro says PASS
    """
    evaluation = macro.evaluate_buy(ticker, ctx)

    flags = " | ".join(evaluation.warning_flags())

    if evaluation.action == FilterAction.SKIP:
        return "SKIP", evaluation.reason

    if evaluation.action == FilterAction.DOWNGRADE:
        new_tier = "MEDIUM" if confidence_tier == "HIGH" else confidence_tier
        updated_reason = f"{reason} | {flags}" if flags else reason
        return new_tier, updated_reason

    # PASS — append any mild flags if present, otherwise leave reason clean
    updated_reason = f"{reason} | {flags}" if flags else reason
    return confidence_tier, updated_reason


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Standalone test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    market = sys.argv[1].upper() if len(sys.argv) > 1 else "INDIA"

    if market not in MARKET_CONFIGS:
        print(f"Unknown market '{market}'. Available: {list(MARKET_CONFIGS.keys())}")
        sys.exit(1)

    print(f"\nRunning macro filter test for: {market}")
    macro   = MacroFilter(MARKET_CONFIGS[market])
    context = macro.run()
    macro.print_summary(context)

    test_tickers = {
        "INDIA":     ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        "AUSTRALIA": ["BHP.AX",      "CBA.AX", "CSL.AX"],
        "CANADA":    ["RY.TO",       "TD.TO",  "CNR.TO"],
        "USA":       ["AAPL",        "MSFT",   "NVDA"],
    }

    for ticker in test_tickers.get(market, []):
        result = macro.evaluate_buy(ticker, context)
        icon   = {"PASS": "✅", "DOWNGRADE": "⚠️", "SKIP": "⛔"}[result.action.value]
        print(f"  {icon} {ticker:16} → {result.action.value:10}  {result.reason}")