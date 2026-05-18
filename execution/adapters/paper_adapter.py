"""
execution/adapters/paper_adapter.py
────────────────────────────────────
Paper trading adapter — logs only, no real orders.
This is the DEFAULT and SAFE mode.
"""

from pathlib import Path
from typing import Dict

from execution.adapters.base import ExecutionAdapter, OrderResult, PortfolioSnapshot
from autopilot.logger import load_portfolio, record_transaction
from ingestion.data_ingestion import fetch_historical_data, get_stock_data


class PaperExecutionAdapter(ExecutionAdapter):
    """Simulates execution by logging to portfolio.json."""

    def place_market_order(self, ticker: str, qty: int, side: str, 
                          tag: str = "T_Raider", price: float = 0.0) -> OrderResult:
        """
        Paper trade: logs to portfolio.json via record_transaction.
        'price' must be passed (LTP) so P&L calculates correctly.
        """
        try:
            record_transaction(
                ticker=ticker,
                side=side.lower(),
                qty=qty,
                price=price,
                strategy_name=f"Paper-{tag}"
            )
            return OrderResult(
                success=True,
                order_id=f"PAPER-{ticker}-{side}",
                status="COMPLETE",
                filled_qty=qty,
                avg_price=price,
                message=f"Paper {side} {qty} shares of {ticker} @ ₹{price:.2f}",
                raw_response={}
            )
        except Exception as e:
            return OrderResult(
                success=False,
                order_id="",
                status="REJECTED",
                filled_qty=0,
                avg_price=0.0,
                message=str(e),
                raw_response={}
            )

    def get_portfolio_snapshot(self, internal_holdings: Dict = None) -> PortfolioSnapshot:
        portfolio = load_portfolio()
        cash = portfolio.get('cash', 0)
        holdings = portfolio.get('holdings', {})

        if not holdings:
            return PortfolioSnapshot(cash=cash, market_value=0.0, total_value=cash, holdings={})

        tickers = list(holdings.keys())
        try:
            market_data = fetch_historical_data(tickers, period="5d")
        except Exception:
            return PortfolioSnapshot(cash=cash, market_value=0.0, total_value=cash, holdings=holdings)

        market_value = 0.0
        for ticker, holding_data in holdings.items():
            qty = self._extract_qty(holding_data)
            df = get_stock_data(market_data, ticker)
            if df.empty:
                continue
            price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            price = float(df[price_col].iloc[-1])
            market_value += qty * price

        return PortfolioSnapshot(
            cash=cash,
            market_value=market_value,
            total_value=cash + market_value,
            holdings=holdings
        )

    def get_ltp(self, ticker: str) -> float:
        try:
            df = fetch_historical_data([ticker], period="1d")
            data = get_stock_data(df, ticker)
            if data.empty:
                return 0.0
            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            return float(data[price_col].iloc[-1])
        except Exception:
            return 0.0

    def is_order_complete(self, order_id: str) -> bool:
        return True  # Paper orders are instant

    @staticmethod
    def _extract_qty(holding_data) -> int:
        if isinstance(holding_data, dict):
            return holding_data.get('qty', 0)
        return int(holding_data or 0)