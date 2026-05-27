"""
execution/adapters/upstox_adapter.py
─────────────────────────────────────
Production adapter for Upstox API v2.
Supports both SANDBOX and LIVE modes via broker_config.json.

Fixes applied:
  FIX 1 — _fetch_upstox_ltp() implemented via MarketQuoteV3Api.get_ltp()
  FIX 2 — yfinance LTP fallback uses .squeeze() to avoid Series→float error
  FIX 3 — instrument_token now resolved via upstox_instrument_cache.json
           (built by instrument_mapper.py) instead of symbol-based guessing.
           Cache format: {"RELIANCE": "NSE_EQ|INE002A01018"}
  FIX 4 — OrderResult() includes all required fields: filled_qty, avg_price
  FIX 5 — is_order_complete() properly implemented (was abstract/missing)
  FIX 6 — Removed hardcoded _SYMBOL_TO_ISIN table — cache covers everything
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import upstox_client
from upstox_client.rest import ApiException

from execution.adapters.base import ExecutionAdapter, OrderResult, PortfolioSnapshot
from execution.instrument_mapper import InstrumentMapper

# Path must match instrument_mapper.py output location
_INSTRUMENT_CACHE_PATH = Path("config/upstox_instrument_cache.json")


class UpstoxExecutionAdapter(ExecutionAdapter):
    """
    Upstox broker adapter using official upstox-python-sdk.
    Set sandbox=true in broker_config.json for testing.
    """

    def __init__(self, config_path: Path = Path("config/broker_config.json")):
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.sandbox      = self.cfg.get('sandbox', True)
        self.access_token = self.cfg.get('access_token', '')
        self.api_version  = "2.0"

        # SDK configuration
        self.configuration = upstox_client.Configuration(sandbox=self.sandbox)
        self.configuration.access_token = self.access_token

        self.api_client    = upstox_client.ApiClient(self.configuration)
        self.order_api     = upstox_client.OrderApi(self.api_client)
        self.portfolio_api = upstox_client.PortfolioApi(self.api_client)
        self.market_api    = upstox_client.MarketQuoteApi(self.api_client)
        self.market_v3_api = upstox_client.MarketQuoteV3Api(self.api_client)
        self.user_api      = upstox_client.UserApi(self.api_client)

        # InstrumentMapper object (secondary fallback)
        self.mapper = InstrumentMapper()

        # FIX 3: Load instrument cache built by instrument_mapper.py
        # Format: {"RELIANCE": "NSE_EQ|INE002A01018", ...}
        self._instrument_cache: Dict[str, str] = self._load_instrument_cache()
        print(f"  ✅ Instrument cache loaded: {len(self._instrument_cache)} symbols")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1 — INSTRUMENT CACHE & TOKEN RESOLUTION
    # ═════════════════════════════════════════════════════════════════════════

    def _load_instrument_cache(self) -> Dict[str, str]:
        """
        Load upstox_instrument_cache.json built by instrument_mapper.py.
        Returns {symbol: instrument_key} e.g. {"RELIANCE": "NSE_EQ|INE002A01018"}
        Warns and returns empty dict if file missing — run instrument_mapper.py.
        """
        if not _INSTRUMENT_CACHE_PATH.exists():
            warnings.warn(
                "upstox_instrument_cache.json not found. "
                "Run: python execution/instrument_mapper.py  to build it."
            )
            return {}
        try:
            with open(_INSTRUMENT_CACHE_PATH) as f:
                cache = json.load(f)
            return cache
        except Exception as e:
            warnings.warn(f"Failed to load instrument cache: {e}")
            return {}

    def _symbol_from_ticker(self, ticker: str) -> str:
        """
        Strip exchange suffix to get plain NSE symbol.
        'RELIANCE.NS' → 'RELIANCE'
        'RELIANCE'    → 'RELIANCE'
        """
        return ticker.replace('.NS', '').replace('.BO', '').upper()

    def _get_instrument_token(self, ticker: str) -> str:
        """
        Convert any ticker format to Upstox instrument_key.

        Valid format : 'NSE_EQ|INE002A01018'  (segment|ISIN)
        Invalid      : 'NSE_EQ|RELIANCE'       (segment|symbol — rejected by API)

        Lookup order:
          1. Already in Upstox format (contains '|') → return as-is
          2. upstox_instrument_cache.json            → direct lookup (primary)
          3. InstrumentMapper object                 → secondary fallback
          4. Warn and return best-guess              → will likely be rejected
        """
        # Already in correct Upstox format
        if '|' in ticker:
            return ticker

        symbol = self._symbol_from_ticker(ticker)

        # Source 1: instrument cache (built by instrument_mapper.py)
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        # Source 2: InstrumentMapper object fallback
        try:
            key = self.mapper.get_instrument_key(symbol)
            if key and '|' in key:
                # Save into session cache to avoid repeated lookups
                self._instrument_cache[symbol] = key
                return key
        except Exception:
            pass

        # Source 3: warn loudly and return best-guess
        warnings.warn(
            f"'{symbol}' not found in instrument cache. "
            f"Run: python execution/instrument_mapper.py  to rebuild the cache. "
            f"Order for {symbol} will likely be rejected by Upstox."
        )
        return f"NSE_EQ|{symbol}"   # best-guess — surfaces a clear API error

    def _upstox_to_yf_ticker(self, ticker: str) -> str:
        """
        Convert Upstox instrument key back to yfinance .NS ticker.
        Uses reverse lookup on the instrument cache.

        'NSE_EQ|INE002A01018' → 'RELIANCE.NS'  (via cache reverse lookup)
        'RELIANCE.NS'         → 'RELIANCE.NS'   (unchanged)
        'RELIANCE'            → 'RELIANCE.NS'   (suffix added)
        """
        if '|' not in ticker:
            symbol = self._symbol_from_ticker(ticker)
            return f"{symbol}.NS"

        # Reverse lookup: find symbol whose cache value matches this key
        for symbol, key in self._instrument_cache.items():
            if key == ticker:
                return f"{symbol}.NS"

        # Cache miss — can't recover symbol from ISIN without a lookup
        warnings.warn(
            f"Reverse instrument lookup failed for '{ticker}' — "
            f"yfinance fallback may fail. Rebuild cache via instrument_mapper.py."
        )
        return ticker   # surfaces a clear error downstream

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2 — LTP (LAST TRADED PRICE)
    # ═════════════════════════════════════════════════════════════════════════

    def _fetch_upstox_ltp(self, ticker: str) -> float:
        """
        FIX 1: Fetch LTP via MarketQuoteV3Api.get_ltp().
        instrument_key must be in 'NSE_EQ|ISIN' format.
        Raises ApiException if blocked (e.g. sandbox) — caller handles fallback.

        Response structure:
            response.data = {"NSE_EQ|INE002A01018": LtpData(ltp=1485.30, ...)}
        """
        instrument_key = self._get_instrument_token(ticker)
        response = self.market_v3_api.get_ltp(instrument_key=instrument_key)

        if response and hasattr(response, 'data') and response.data:
            data = response.data
            if isinstance(data, dict):
                # Try exact key match first, then first available value
                entry = data.get(instrument_key) or next(iter(data.values()), None)
                if entry and hasattr(entry, 'ltp'):
                    return float(entry.ltp)

        return 0.0

    def _fetch_yfinance_ltp(self, ticker: str) -> float:
        """
        FIX 2: yfinance LTP fallback.
        Uses .squeeze() before .iloc[-1] to handle newer yfinance versions
        which return a single-element Series instead of a scalar for
        single-ticker downloads.
        """
        import yfinance as yf

        yf_ticker = self._upstox_to_yf_ticker(ticker)
        if not yf_ticker.endswith('.NS') and not yf_ticker.endswith('.BO'):
            yf_ticker = f"{yf_ticker}.NS"

        df = yf.download(yf_ticker, period="2d", progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"yfinance returned empty DataFrame for {yf_ticker}")

        # .squeeze() collapses single-column DataFrame to Series,
        # then .iloc[-1] gives a scalar float
        close = df["Close"].squeeze()
        return float(close.iloc[-1])

    def get_ltp(self, ticker: str) -> float:
        """
        Get Last Traded Price with automatic yfinance fallback.
        Upstox sandbox blocks the market data API so fallback is always needed
        in sandbox mode.
        """
        # Tier 1: Upstox MarketQuoteV3Api
        try:
            ltp = self._fetch_upstox_ltp(ticker)
            if ltp > 0:
                return ltp
            print(f"  ⚠️  Upstox LTP returned 0 for {ticker} — trying yfinance")
        except Exception as e:
            print(f"  ⚠️  Upstox LTP blocked for {ticker} ({e}) — yfinance fallback")

        # Tier 2: yfinance
        try:
            ltp = self._fetch_yfinance_ltp(ticker)
            if ltp > 0:
                print(f"  ✅ yfinance LTP for {ticker}: ₹{ltp:.2f}")
                return ltp
        except Exception as e:
            print(f"  ⚠️  yfinance LTP also failed for {ticker}: {e}")

        return 0.0

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3 — ORDER PLACEMENT
    # ═════════════════════════════════════════════════════════════════════════

    def place_market_order(
        self,
        ticker: str,
        qty: int,
        side: str,
        tag: str = "",
        price: float = 0.0,
    ) -> OrderResult:
        """
        Place a market order via Upstox OrderApi.
        FIX 3: instrument_token resolved via cache (ISIN-based key).
        FIX 4: OrderResult includes all required fields (filled_qty, avg_price).
        """
        try:
            instrument_token = self._get_instrument_token(ticker)

            # Get LTP for reference price (not used in MARKET orders but
            # useful for logging and post-order reconciliation)
            ltp = price if price > 0 else self.get_ltp(ticker)

            body = upstox_client.PlaceOrderRequest(
                quantity=qty,
                product="D",                    # D = Delivery / CNC
                validity="DAY",
                price=0.0,                      # 0.0 required for MARKET orders
                tag=tag or "T_Raider",
                instrument_token=instrument_token,
                order_type="MARKET",
                transaction_type=side.upper(),  # "BUY" or "SELL"
                disclosed_quantity=0,
                trigger_price=0.0,
                is_amo=False,
            )

            api_response = self.order_api.place_order(body, self.api_version)
            order_id = str(api_response.data.order_id)
            print(
                f"  ✅ Order placed: {side.upper()} {qty} × {ticker} "
                f"@ ₹{ltp:.2f} (MARKET) | ID: {order_id}"
            )

            return OrderResult(
                success=True,
                order_id=order_id,
                status="OPEN",
                message="Order placed successfully",
                filled_qty=0,     # not yet filled at placement time
                avg_price=0.0,    # not yet filled at placement time
            )

        except Exception as e:
            print(f"  ❌ Order failed for {ticker}: {e}")
            return OrderResult(
                success=False,
                order_id="",
                status="FAILED",
                message=str(e),
                filled_qty=0,
                avg_price=0.0,
            )

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 4 — ORDER STATUS
    # ═════════════════════════════════════════════════════════════════════════

    def _get_order_status(self, order_id: str) -> str:
        """
        Fetch order status string from Upstox.
        Returns lowercase status string or 'unknown' on failure.
        """
        try:
            resp = self.order_api.get_order_details(
                order_id=order_id,
                api_version=self.api_version
            )
            data = resp.data if hasattr(resp, 'data') else resp
            return getattr(data, 'status', 'unknown').lower()
        except Exception:
            return 'unknown'

    def is_order_complete(self, order_id: str) -> bool:
        """
        FIX 5: Properly implemented abstract method from ExecutionAdapter base.
        Returns True if order reached a terminal state.
        Defaults to True on exception since sandbox blocks this API.
        """
        try:
            status = self._get_order_status(order_id)
            return status in ("complete", "cancelled", "rejected")
        except Exception:
            # Sandbox blocks order status API — treat as complete to avoid hangs
            return True

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5 — PORTFOLIO SNAPSHOT
    # ═════════════════════════════════════════════════════════════════════════

    def _load_paper_portfolio(self) -> PortfolioSnapshot:
        """
        Sandbox fallback: read position data from portfolio.json and fetch
        live prices via yfinance to compute current market value.
        Used when Upstox sandbox blocks the holdings/funds API.
        """
        try:
            with open("config/portfolio.json") as f:
                p = json.load(f)
        except Exception:
            return PortfolioSnapshot(
                cash=0, market_value=0, total_value=0, holdings={}
            )

        cash     = float(p.get("cash", 0))
        holdings = p.get("holdings", {})

        market_value = 0.0
        for ticker, data in holdings.items():
            qty = data.get("qty", 0) if isinstance(data, dict) else int(data or 0)
            if qty <= 0:
                continue
            try:
                ltp = self._fetch_yfinance_ltp(ticker)
            except Exception:
                ltp = float(
                    data.get("entry_price", 0) if isinstance(data, dict) else 0
                )
            market_value += qty * ltp

        return PortfolioSnapshot(
            cash=cash,
            market_value=round(market_value, 2),
            total_value=round(cash + market_value, 2),
            holdings=holdings,
        )

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """
        Fetch live holdings and funds from Upstox.
        Falls back to portfolio.json when sandbox blocks the broker API.
        """
        try:
            # Funds / available cash
            funds_resp = self.user_api.get_user_fund_margin(self.api_version)
            funds_data = funds_resp.data if hasattr(funds_resp, 'data') else funds_resp
            cash = float(
                getattr(
                    funds_data,
                    'available_opening_balance',
                    getattr(funds_data, 'available_margin', 0)
                )
            )

            # Holdings
            holdings_resp = self.portfolio_api.get_holdings(self.api_version)
            holdings_data = (
                holdings_resp.data
                if hasattr(holdings_resp, 'data')
                else holdings_resp
            )

            broker_holdings: Dict = {}
            market_value = 0.0

            if hasattr(holdings_data, '__iter__'):
                for h in holdings_data:
                    sym = getattr(
                        h, 'trading_symbol',
                        getattr(h, 'tradingsymbol', '')
                    )
                    qty = int(getattr(h, 'quantity', 0))
                    avg = float(getattr(h, 'average_price', 0))
                    ltp = float(getattr(h, 'last_price', avg))

                    broker_holdings[sym] = {"qty": qty, "entry_price": avg}
                    market_value += qty * ltp

            if cash == 0 and not broker_holdings:
                raise ValueError(
                    "Broker returned empty portfolio — likely sandbox limitation"
                )

            return PortfolioSnapshot(
                cash=cash,
                market_value=round(market_value, 2),
                total_value=round(cash + market_value, 2),
                holdings=broker_holdings,
            )

        except Exception as e:
            print(f"  ⚠️  Broker portfolio fetch failed: {e}")
            print("  ℹ️   Sandbox fallback: reading positions from portfolio.json")
            return self._load_paper_portfolio()