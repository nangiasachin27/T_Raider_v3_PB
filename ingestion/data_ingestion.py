"""
ingestion/data_ingestion.py — ENHANCED with multi-source fallback
─────────────────────────────────────────────────────────────────
Fetches historical stock data with automatic fallback:
  1. yfinance (primary)
  2. NSE Bhavcopy (fallback — stub)
  3. Cached data with staleness warning (last resort)

Usage:
    from ingestion.data_ingestion import fetch_historical_data
    data = fetch_historical_data(["RELIANCE.NS", "TCS.NS"], period="5d")
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ── Path fix ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path("config/.data_cache")
CACHE_DIR.mkdir(exist_ok=True)
MAX_CACHE_AGE_DAYS = 7

# ═════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═════════════════════════════════════════════════════════════════════════════

class DataOutageError(Exception):
    """Raised when all data sources fail."""
    pass

class StaleDataWarning(UserWarning):
    """Warning when using cached stale data."""
    pass

# ═════════════════════════════════════════════════════════════════════════════
# CACHE MANAGEMENT — Pickle-based (preserves MultiIndex)
# ═════════════════════════════════════════════════════════════════════════════

def _cache_key(tickers: List[str], period: str) -> str:
    """Generate cache filename from tickers and period."""
    tickers_hash = "_".join(sorted(tickers))[:50]
    return f"cache_{tickers_hash}_{period}_{datetime.now().strftime('%Y%m%d')}"

def _save_to_cache(data: pd.DataFrame, tickers: List[str], period: str):
    """Save fetched data to local cache using pickle (preserves MultiIndex)."""
    if data.empty:
        return
    cache_path = CACHE_DIR / (_cache_key(tickers, period) + ".pkl")
    data.to_pickle(cache_path)

def _load_from_cache(tickers: List[str], period: str) -> Optional[pd.DataFrame]:
    """Load data from cache if available and not too old."""
    cache_path = CACHE_DIR / (_cache_key(tickers, period) + ".pkl")
    if not cache_path.exists():
        return None
    
    age_days = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days
    if age_days > MAX_CACHE_AGE_DAYS:
        return None
    
    try:
        return pd.read_pickle(cache_path)
    except Exception:
        return None

def _cleanup_old_cache():
    """Remove cache files older than MAX_CACHE_AGE_DAYS."""
    for f in CACHE_DIR.glob("*.pkl"):
        age_days = (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days
        if age_days > MAX_CACHE_AGE_DAYS:
            f.unlink()

# ═════════════════════════════════════════════════════════════════════════════
# DATA FRESHNESS VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def _count_trading_days(start_date, end_date) -> int:
    """Count trading days (Mon-Fri) between dates."""
    count = 0
    current = start_date + timedelta(days=1)
    while current <= end_date:
        if current.weekday() < 5:
            count += 1
        current += timedelta(days=1)
    return count

def validate_data_freshness(data: pd.DataFrame, max_trading_days_stale: int = 1) -> Tuple[bool, str]:
    """Check if data is fresh enough for trading."""
    if data.empty:
        return False, "Empty dataframe"
    
    last_date = data.index[-1]
    if hasattr(last_date, 'date'):
        last_date = last_date.date()
    
    today = datetime.now().date()
    trading_days_stale = _count_trading_days(last_date, today)
    
    if trading_days_stale > max_trading_days_stale:
        return False, f"Data is {trading_days_stale} trading days stale (last: {last_date})"
    
    return True, f"Fresh (last: {last_date}, {trading_days_stale} trading days ago)"

# ═════════════════════════════════════════════════════════════════════════════
# PRIMARY SOURCE: yfinance
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_yfinance(tickers: List[str], period: str = "5d") -> Optional[pd.DataFrame]:
    """Fetch data from yfinance. Preserves MultiIndex columns."""
    try:
        data = yf.download(tickers, period=period, progress=False)
        if data.empty:
            return None
        # DON'T flatten columns — keep MultiIndex (Price, Ticker) for uniqueness
        return data
    except Exception as e:
        warnings.warn(f"yfinance fetch failed: {e}")
        return None

# ═════════════════════════════════════════════════════════════════════════════
# FALLBACK SOURCE: NSE Bhavcopy (stub)
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_bhavcopy(tickers: List[str], period: str = "5d") -> Optional[pd.DataFrame]:
    """NSE Bhavcopy fallback — not fully implemented."""
    warnings.warn("NSE Bhavcopy fallback not fully implemented.")
    return None

# ═════════════════════════════════════════════════════════════════════════════
# LAST RESORT: Cached data with warning
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_cached(tickers: List[str], period: str) -> Optional[pd.DataFrame]:
    """Load from cache with explicit staleness warning."""
    data = _load_from_cache(tickers, period)
    if data is not None:
        warnings.warn(
            f"Using cached data for {tickers[0]}... (may be stale). "
            "Trading accuracy may be reduced.",
            StaleDataWarning
        )
    return data

# ═════════════════════════════════════════════════════════════════════════════
# MAIN PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def fetch_historical_data(
    tickers: List[str],
    period: str = "5d",
    max_stale_days: int = 1,
    allow_cache_fallback: bool = True
) -> pd.DataFrame:
    """
    Fetch historical data with automatic multi-source fallback.
    
    Args:
        tickers: List of ticker symbols
        period: yfinance period string ("5d", "1mo", "3mo", etc.)
        max_stale_days: Maximum allowed trading days stale
        allow_cache_fallback: Whether to use cached data as last resort
    
    Returns:
        DataFrame with OHLCV data (MultiIndex columns for multi-ticker)
    
    Raises:
        DataOutageError: If all sources fail and cache not allowed/available
    """
    print(f"📊 Fetching {period} data for {len(tickers)} tickers...")
    # ── Source 1: yfinance ────────────────────────────────────────────────
    data = _fetch_yfinance(tickers, period)
    if data is not None:
        is_fresh, reason = validate_data_freshness(data, max_stale_days)
        if is_fresh:
            print("✅ [yfinance] Data fetched successfully.")
            _save_to_cache(data, tickers, period)
            _cleanup_old_cache()
            return data
        else:
            print(f"⚠️ [yfinance] Stale: {reason}")
    
    # ── Source 2: NSE Bhavcopy (India only) ───────────────────────────────
    data = _fetch_bhavcopy(tickers, period)
    if data is not None:
        is_fresh, reason = validate_data_freshness(data, max_stale_days)
        if is_fresh:
            print("✅ [NSE Bhavcopy] Fallback data fetched successfully.")
            _save_to_cache(data, tickers, period)
            return data
        else:
            print(f"⚠️ [Bhavcopy] Stale: {reason}")
    
    # ── Source 3: Cache (last resort) ─────────────────────────────────────
    if allow_cache_fallback:
        data = _fetch_cached(tickers, period)
        if data is not None:
            print("⚠️ [CACHE] Using cached data — accuracy may be reduced.")
            return data
    
    # ── All sources failed ────────────────────────────────────────────────
    raise DataOutageError(
        f"All data sources failed for {len(tickers)} tickers. "
        f"yfinance: unavailable/stale, bhavcopy: not implemented, "
        f"cache: {'not available' if not allow_cache_fallback else 'exhausted'}."
    )

# ═════════════════════════════════════════════════════════════════════════════
# HELPER: Single stock data extraction (handles MultiIndex)
# ═════════════════════════════════════════════════════════════════════════════

def get_stock_data(full_market_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract single stock data from multi-ticker dataframe."""
    if full_market_data.empty:
        return pd.DataFrame()
    
    # Handle MultiIndex columns (new yfinance format)
    if isinstance(full_market_data.columns, pd.MultiIndex):
        if ticker in full_market_data.columns.get_level_values(1):
            df = full_market_data.xs(ticker, level=1, axis=1)
            return df
        return pd.DataFrame()
    else:
        # Old format: single ticker or flat columns
        if ticker in full_market_data.columns:
            return full_market_data[[ticker]].copy()
        return pd.DataFrame()

# ═════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    try:
        data = fetch_historical_data(tickers, period="5d")
        print(f"\nShape: {data.shape}")
        print(f"Columns: {list(data.columns[:5])}...")
        print(f"Last date: {data.index[-1]}")
        print(f"\nSample (last 2 rows):")
        print(data.tail(2))
    except DataOutageError as e:
        print(f"FAILED: {e}")