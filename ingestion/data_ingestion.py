import yfinance as yf
import pandas as pd
import time
import random
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# jugaad-data is an optional dependency. If it isn't installed the Bhav Copy
# fallback is silently skipped and the caller sees an empty DataFrame (same
# behaviour as before).  Install with:  pip install jugaad-data
# ---------------------------------------------------------------------------
try:
    from jugaad_data.nse import bhavcopy_load
    _JUGAAD_AVAILABLE = True
except ImportError:
    _JUGAAD_AVAILABLE = False


# ── helpers ──────────────────────────────────────────────────────────────────

def _period_to_dates(period: str):
    """
    Converts a yfinance-style period string ('2y', '5y', '1mo', '6mo' …)
    into (start_date, end_date) as datetime.date objects.
    Supported suffixes: d (days), mo (months), y (years).
    """
    today = date.today()
    period = period.lower().strip()

    if period.endswith('y'):
        years = int(period[:-1])
        start = today.replace(year=today.year - years)
    elif period.endswith('mo'):
        months = int(period[:-2])
        year  = today.year  - (months // 12)
        month = today.month - (months %  12)
        if month <= 0:
            month += 12
            year  -= 1
        start = today.replace(year=year, month=month)
    elif period.endswith('d'):
        start = today - timedelta(days=int(period[:-1]))
    else:
        # Unrecognised format — default to 2 years
        start = today.replace(year=today.year - 2)

    return start, today


def _nse_symbol(ticker: str) -> str:
    """Strips the '.NS' / '.BO' suffix to get the bare NSE symbol."""
    return ticker.split('.')[0]


def _trading_days(start: date, end: date):
    """
    Returns a list of weekday dates between start and end (inclusive).
    Does NOT exclude NSE holidays — bhavcopy_load handles that by returning
    None for non-trading days, which we simply skip.
    """
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:   # Mon–Fri
            days.append(current)
        current += timedelta(days=1)
    return days


# ── Bhav Copy fetcher ─────────────────────────────────────────────────────────

def _fetch_via_bhavcopy(tickers: list, period: str = "2y") -> pd.DataFrame:
    """
    Downloads NSE EOD data using jugaad-data's bhavcopy_load().

    Returns a MultiIndex DataFrame that matches yfinance's output format:
        columns = MultiIndex[(Open, TICKER), (High, TICKER), …]
    so that get_stock_data() and all callers work without modification.

    Bhav Copy does NOT provide an Adjusted Close, so the 'Adj Close' column
    is set equal to 'Close' (raw close price). Corporate-action adjustments
    are absent — acceptable for EOD signal generation but worth noting.

    Rate-limit note: NSE archive servers throttle aggressive polling. A
    random 0.5–1.5 s sleep between daily requests is baked in. Fetching 5
    years (~1250 trading days) takes roughly 15–30 minutes.
    """
    if not _JUGAAD_AVAILABLE:
        raise ImportError("jugaad-data is not installed. Run: pip install jugaad-data")

    start, end = _period_to_dates(period)
    days        = _trading_days(start, end)
    bare_syms   = [_nse_symbol(t) for t in tickers]

    print(f"📥 [Bhav Copy] Fetching {period} of data ({len(days)} trading days) …")

    # Accumulate per-day DataFrames
    daily_frames = []

    for trading_date in days:
        try:
            df = bhavcopy_load(trading_date)   # returns None on holidays/weekends
            if df is None or df.empty:
                continue

            # Normalise column names — jugaad-data uses uppercase
            df.columns = [c.strip().upper() for c in df.columns]

            # Keep only the tickers we care about
            # 'SYMBOL' column holds bare NSE names e.g. 'RELIANCE'
            if 'SYMBOL' not in df.columns:
                continue

            df = df[df['SYMBOL'].isin(bare_syms)].copy()
            if df.empty:
                continue

            df['Date'] = pd.to_datetime(trading_date)
            df = df.set_index('Date')

            # Rename to yfinance-compatible column names
            rename = {
                'OPEN':      'Open',
                'HIGH':      'High',
                'LOW':       'Low',
                'CLOSE':     'Close',
                'TOTTRDQTY': 'Volume',   # jugaad-data column name
            }
            df = df.rename(columns=rename)

            # Add Adj Close = Close (no corporate-action data available)
            df['Adj Close'] = df['Close']

            # Keep only OHLCV + Adj Close + SYMBOL
            keep = ['SYMBOL', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            df = df[[c for c in keep if c in df.columns]]

            daily_frames.append(df)

        except Exception:
            # Silently skip days where the archive file isn't available
            pass

        # Polite delay to avoid hammering NSE servers
        time.sleep(random.uniform(0.5, 1.5))

    if not daily_frames:
        raise ValueError("Bhav Copy returned no data for the requested period.")

    combined = pd.concat(daily_frames)

    # ── Pivot into MultiIndex to match yfinance output ────────────────────
    # Target shape:
    #   index  = Date
    #   columns = MultiIndex[(Open, TICKER.NS), (High, TICKER.NS), …]

    # Map bare symbol → full ticker (e.g. RELIANCE → RELIANCE.NS)
    sym_to_ticker = {_nse_symbol(t): t for t in tickers}

    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    pieces = []

    for sym, ticker in sym_to_ticker.items():
        sub = combined[combined['SYMBOL'] == sym][ohlcv_cols].copy()
        if sub.empty:
            continue
        sub.columns = pd.MultiIndex.from_product([[ticker], sub.columns])
        pieces.append(sub)

    if not pieces:
        raise ValueError("No matching symbols found in Bhav Copy data.")

    result = pd.concat(pieces, axis=1).sort_index()
    result = result.dropna(how='all')

    # Swap MultiIndex levels to match yfinance: (field, ticker) not (ticker, field)
    result.columns = result.columns.swaplevel(0, 1)
    result.sort_index(axis=1, level=0, inplace=True)

    print(f"✅ [Bhav Copy] Data assembled for {len(pieces)} stocks.\n")
    return result


# ── Public API (drop-in replacements for the originals) ──────────────────────

def fetch_historical_data(tickers: list, period: str = "2y") -> pd.DataFrame:
    """
    Fetches OHLCV data with automatic fallback:
      1. yfinance  (fast, may be blocked by Yahoo rate-limiting)
      2. NSE Bhav Copy via jugaad-data  (slower, but direct from NSE archive)

    Returns a MultiIndex DataFrame identical in shape to yfinance's output so
    that get_stock_data() and all callers work without any changes.
    """
    print(f"📥 Fetching {period} of data for {len(tickers)} stocks…")

    # ── Layer 1: yfinance ─────────────────────────────────────────────────
    try:
        data = yf.download(
            tickers,
            period=period,
            progress=False,
            group_by='ticker',
        )
        if not data.empty:
            print("✅ [yfinance] Data fetched successfully.\n")
            return data
        else:
            print("⚠️  [yfinance] Empty response — trying Bhav Copy fallback…")
    except Exception as e:
        print(f"⚠️  [yfinance] Failed ({e}) — trying Bhav Copy fallback…")

    # ── Layer 2: NSE Bhav Copy ────────────────────────────────────────────
    if not _JUGAAD_AVAILABLE:
        print("❌ jugaad-data not installed. Install with: pip install jugaad-data")
        print("   Both data sources failed. Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        data = _fetch_via_bhavcopy(tickers, period)
        return data
    except Exception as e:
        print(f"❌ [Bhav Copy] Also failed: {e}")
        print("   Both data sources exhausted. Returning empty DataFrame.")
        return pd.DataFrame()


def get_stock_data(full_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Extracts the OHLCV slice for a single ticker from the combined DataFrame.
    Unchanged from the original — works with both yfinance and Bhav Copy output.
    """
    try:
        if isinstance(full_df.columns, pd.MultiIndex):
            return full_df[ticker].dropna()
        else:
            return full_df.dropna()
    except KeyError:
        return pd.DataFrame()