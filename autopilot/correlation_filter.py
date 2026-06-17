"""
Correlation Filter Module — Prevent concentration risk
─────────────────────────────────────────────────────
Rejects buy signals if new ticker correlates too highly
with any existing portfolio holding.

Usage:
    from daily_screener import CorrelationFilter
    ok, reason = CorrelationFilter.check(ticker, portfolio, market_data)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from pathlib import Path

class CorrelationFilter:
    """
    Prevents buying stocks that are too correlated with existing holdings.
    Reduces sector concentration and systematic risk.
    """

    # Hardcoded thresholds — edit code to change
    DEFAULT_THRESHOLD = 0.82      # Reject if |correlation| > 0.75
    MIN_HISTORY_DAYS = 30         # Need 30 days of overlapping data
    LOOKBACK_DAYS = 90            # Use last 90 days for correlation

    @classmethod
    def check(
        cls,
        ticker: str,
        portfolio: Dict,
        full_market_data: pd.DataFrame,
        threshold: float = None
    ) -> Tuple[bool, str]:
        """
        Check if ticker passes correlation filter.

        Returns:
            (passed: bool, reason: str)
        """
        if threshold is None:
            threshold = cls.DEFAULT_THRESHOLD

        holdings = portfolio.get("holdings", {})

        # No holdings = no correlation risk
        if not holdings:
            return True, "No existing holdings"

        # Extract new ticker returns
        new_returns = cls._get_returns(full_market_data, ticker)
        if new_returns is None:
            return False, f"{ticker}: insufficient data for correlation"

        # Check against each holding
        correlations = []

        for held_ticker in holdings.keys():
            held_returns = cls._get_returns(full_market_data, held_ticker)
            if held_returns is None:
                continue

            # Align and compute correlation
            corr = cls._compute_correlation(new_returns, held_returns)
            if corr is None:
                continue

            correlations.append((held_ticker, corr))

            # Early exit if any correlation exceeds threshold
            if abs(corr) > threshold:
                return False, (
                    f"{ticker} correlates {corr:+.2f} with {held_ticker} "
                    f"(threshold: ±{threshold})"
                )

        # All correlations below threshold
        if correlations:
            max_corr = max(abs(c) for _, c in correlations)
            max_ticker = [t for t, c in correlations if abs(c) == max_corr][0]
            return True, (
                f"Max correlation: {max_corr:.2f} with {max_ticker} "
                f"(below threshold {threshold})"
            )

        return True, "No comparable data for correlation check"

    @classmethod
    def _get_returns(cls, full_market_data: pd.DataFrame, ticker: str) -> pd.Series:
        """Extract daily returns for a ticker."""
        from ingestion.data_ingestion import get_stock_data

        df = get_stock_data(full_market_data, ticker)
        if df.empty:
            return None

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Get Close or Adj Close
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        if price_col not in df.columns:
            return None

        prices = df[price_col].dropna()
        if len(prices) < cls.MIN_HISTORY_DAYS:
            return None

        # Use last LOOKBACK_DAYS
        prices = prices.tail(cls.LOOKBACK_DAYS)
        returns = prices.pct_change().dropna()

        return returns if len(returns) >= cls.MIN_HISTORY_DAYS // 2 else None

    @classmethod
    def _compute_correlation(
        cls,
        returns1: pd.Series,
        returns2: pd.Series
    ) -> float:
        """Compute Pearson correlation between two return series."""
        # Align by date
        combined = pd.concat([returns1, returns2], axis=1).dropna()

        if len(combined) < cls.MIN_HISTORY_DAYS // 2:
            return None

        corr = combined.corr().iloc[0, 1]
        return corr if not np.isnan(corr) else None

    @classmethod
    def get_portfolio_correlation_matrix(
        cls,
        portfolio: Dict,
        full_market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate correlation matrix for all holdings + candidate tickers.
        Useful for portfolio analysis.
        """
        holdings = list(portfolio.get("holdings", {}).keys())
        if not holdings:
            return pd.DataFrame()

        returns_dict = {}
        for ticker in holdings:
            ret = cls._get_returns(full_market_data, ticker)
            if ret is not None:
                returns_dict[ticker] = ret

        if not returns_dict:
            return pd.DataFrame()

        df = pd.DataFrame(returns_dict)
        return df.corr()


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRATION: Add to daily_screener.py run_screener()
# ═════════════════════════════════════════════════════════════════════════════

def run_screener_with_correlation(
    tickers,
    full_market_data,
    portfolio,
    optimal_params,
    mode="CONSERVATIVE",
    apply_correlation_filter=True
):
    """
    Enhanced screener with optional correlation filter.

    Add this call inside your existing run_screener() after signal generation:

        if apply_correlation_filter and latest_signal == 1:
            corr_ok, corr_reason = CorrelationFilter.check(
                ticker, portfolio, full_market_data
            )
            if not corr_ok:
                print(f"  {ticker}: SKIP — {corr_reason}")
                continue
    """
    # This is a placeholder showing integration point
    # Actual implementation goes in your daily_screener.py
    pass


if __name__ == "__main__":
    # Quick test
    print("CorrelationFilter loaded successfully")
    print(f"Default threshold: ±{CorrelationFilter.DEFAULT_THRESHOLD}")
    print(f"Lookback: {CorrelationFilter.LOOKBACK_DAYS} days")