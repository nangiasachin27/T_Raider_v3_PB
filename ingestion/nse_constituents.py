"""
ingestion/nse_constituents.py
─────────────────────────────
Two responsibilities:

  1. SURVIVORSHIP BIAS MITIGATION
     Works with ANY universe defined in config/stocks.json.
     No hardcoded tickers or index-specific logic.
     Determines each ticker's real data window from actual price history —
     a stock listed in 2021 simply won't have data before 2021, so the first
     available row IS the survivorship-bias-safe start date.
     Used by auto_optimizer.py during universe validation.

  2. MARKET REGIME FILTER
     Checks whether the broad Indian market (Nifty 50 index ^NSEI) is in an
     uptrend or downtrend via a 50-day EMA comparison.
     Used by daily_screener.py to suppress BUY signals in bear markets.
     Always uses ^NSEI as the market barometer regardless of what stocks
     you're trading — whether Nifty 50, Nifty 200, or a custom watchlist.
"""

import json
import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────────────────────────────────────
# Config reader
# ─────────────────────────────────────────────────────────────────────────────

def load_universe(config_path: str = "config/stocks.json") -> List[str]:
    """
    Loads the full ticker universe from stocks.json.

    Reads ALL list-valued keys in the JSON and merges them — so if you add
    a 'nifty_next50' or 'watchlist' key alongside 'nifty_50', all tickers
    are included automatically without any code changes.

    Returns:
        Deduplicated list of tickers preserving order (e.g. ['RELIANCE.NS', ...]).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"stocks.json not found at: {config_path}")

    with open(path) as f:
        config = json.load(f)

    all_tickers = []
    for value in config.values():
        if isinstance(value, list):
            all_tickers.extend(value)

    # Deduplicate while preserving insertion order
    seen, unique = set(), []
    for t in all_tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — SURVIVORSHIP BIAS: Data-Driven Universe Validation
# ─────────────────────────────────────────────────────────────────────────────

def get_ticker_data_window(ticker: str,
                           full_df: pd.DataFrame) -> Tuple[Optional[date], Optional[date], int]:
    """
    Determines the actual date range where a ticker has valid price data
    in the already-downloaded DataFrame.

    This is the data-driven survivorship bias fix:
      - A stock listed in Sep 2020 will have no data before Sep 2020.
      - Clipping the backtest to data_start means we never test it on
        data from before it existed — no hardcoded index membership needed.

    Returns:
        (data_start, data_end, num_rows)
        Returns (None, None, 0) if the ticker has no data.
    """
    try:
        df = full_df[ticker].dropna(how='all') if isinstance(full_df.columns, pd.MultiIndex) \
             else full_df.dropna(how='all')

        if df.empty:
            return None, None, 0

        df.index = pd.to_datetime(df.index)
        return df.index[0].date(), df.index[-1].date(), len(df)

    except KeyError:
        return None, None, 0
    except Exception as e:
        warnings.warn(f"Unexpected error getting data window for {ticker}: {e}")
        return None, None, 0


def validate_universe(tickers: List[str],
                      full_df: pd.DataFrame,
                      min_days: int = 500) -> Dict[str, Dict]:
    """
    Validates every ticker in the universe against the downloaded price data.

    Args:
        tickers  : Full ticker list from stocks.json.
        full_df  : MultiIndex DataFrame from fetch_historical_data().
        min_days : Minimum trading days required for optimization (default 500).
                   Below this threshold a ticker is flagged 'insufficient'
                   and skipped — not enough data for a meaningful WF fold.

    Returns:
        Dict keyed by ticker with keys:
            status     : "ok" | "insufficient" | "no_data"
            data_start : date or None
            data_end   : date or None
            num_rows   : int
            note       : human-readable explanation
    """
    report = {}
    for ticker in tickers:
        data_start, data_end, num_rows = get_ticker_data_window(ticker, full_df)

        if data_start is None:
            report[ticker] = {
                "status":     "no_data",
                "data_start": None,
                "data_end":   None,
                "num_rows":   0,
                "note":       "No data from yfinance — possibly delisted or wrong symbol",
            }
        elif num_rows < min_days:
            report[ticker] = {
                "status":     "insufficient",
                "data_start": data_start,
                "data_end":   data_end,
                "num_rows":   num_rows,
                "note":       f"Only {num_rows} rows — below {min_days}-day minimum for WF validation",
            }
        else:
            folds_est = max(0, (num_rows - 756) // 252)
            report[ticker] = {
                "status":     "ok",
                "data_start": data_start,
                "data_end":   data_end,
                "num_rows":   num_rows,
                "note":       "" if folds_est >= 2 else f"Short history — only ~{folds_est} WF fold(s) possible",
            }

    return report


def print_universe_report(report: Dict[str, Dict]):
    """Prints a clear console summary of the universe validation."""
    ok           = {t: v for t, v in report.items() if v['status'] == 'ok'}
    insufficient = {t: v for t, v in report.items() if v['status'] == 'insufficient'}
    no_data      = {t: v for t, v in report.items() if v['status'] == 'no_data'}

    print(f"\n{'='*70}")
    print(f"Universe Validation: {len(ok)} OK | {len(insufficient)} Insufficient | {len(no_data)} No Data")
    print(f"{'='*70}")

    if no_data:
        print(f"\n❌ NO DATA ({len(no_data)}) — check symbols or remove from stocks.json:")
        for t, v in no_data.items():
            print(f"   {t:22} {v['note']}")

    if insufficient:
        print(f"\n⚠️  INSUFFICIENT HISTORY ({len(insufficient)}) — skipped in optimizer:")
        for t, v in insufficient.items():
            print(f"   {t:22} {v['num_rows']} rows | from {v['data_start']}")

    if ok:
        print(f"\n✅ READY ({len(ok)}):")
        for t, v in ok.items():
            note = f"  ← {v['note']}" if v['note'] else ""
            print(f"   {t:22} {v['num_rows']:5} rows | {v['data_start']} → {v['data_end']}{note}")
    print()


def save_universe_report(report: Dict[str, Dict],
                         path: str = "config/universe_report.json"):
    """Persists the validation report to config/ for audit trail."""
    serializable = {
        ticker: {
            **v,
            "data_start": str(v["data_start"]) if v["data_start"] else None,
            "data_end":   str(v["data_end"])   if v["data_end"]   else None,
        }
        for ticker, v in report.items()
    }
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"📄 Universe report saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — MARKET REGIME FILTER
# ─────────────────────────────────────────────────────────────────────────────

def get_market_regime(lookback_days: int = 150,
                      ema_span: int = 50) -> Tuple[bool, float, float]:
    """
    Checks whether the broad Indian market is in an uptrend or downtrend.

    Always uses ^NSEI (Nifty 50 index) as the barometer — regardless of
    what stocks are in your trading universe. The broad market regime is
    the same signal whether you trade 50 stocks or 200.

    Logic:
        Uptrend   = Nifty close > 50-day EMA  →  BUY signals allowed
        Downtrend = Nifty close < 50-day EMA  →  BUY signals suppressed

    Args:
        lookback_days : Calendar days of index data to fetch.
                        150 days gives ~100 trading days — enough for EMA50.
        ema_span      : EMA window in trading days. Default 50.

    Returns:
        (is_uptrend: bool, latest_close: float, ema_value: float)
        Defaults to (True, 0.0, 0.0) on any failure so a bad data day
        never incorrectly suppresses all signals.
    """
    try:
        nifty = yf.download("^NSEI", period=f"{lookback_days}d",
                            progress=False, auto_adjust=True)

        if nifty.empty or len(nifty) < ema_span:
            warnings.warn("Insufficient Nifty data for regime check — defaulting to uptrend.")
            return True, 0.0, 0.0

        close  = nifty["Close"].squeeze()
        latest = float(close.iloc[-1])
        ema    = float(close.ewm(span=ema_span, adjust=False).mean().iloc[-1])
        return latest > ema, latest, ema

    except Exception as e:
        warnings.warn(f"Regime check failed ({e}) — defaulting to uptrend.")
        return True, 0.0, 0.0


def regime_summary(is_uptrend: bool, latest: float, ema: float,
                   ema_span: int = 50) -> str:
    """Returns a one-line human-readable regime description."""
    direction = "UPTREND ▲" if is_uptrend else "DOWNTREND ▼"
    pct_diff  = ((latest - ema) / ema * 100) if ema > 0 else 0
    sign      = "+" if pct_diff >= 0 else ""
    return (
        f"Nifty 50: {direction}  |  "
        f"Close: {latest:,.0f}  |  EMA{ema_span}: {ema:,.0f}  |  "
        f"Spread: {sign}{pct_diff:.2f}%"
    )