"""
execution/adapters/upstox_adapter.py
────────────────────────────────────
Production adapter for Upstox API v2.
Supports both SANDBOX and LIVE modes via config.

To use:
  1. Set sandbox=true in broker_config.json for testing
  2. Set sandbox=false + live access_token for production
  3. Ensure instrument cache is built: python execution/instrument_mapper.py
"""

import json
import time
from pathlib import Path
from typing import Dict

import upstox_client
from upstox_client.rest import ApiException

from execution.adapters.base import ExecutionAdapter, OrderResult, PortfolioSnapshot
from execution.instrument_mapper import InstrumentMapper


class UpstoxExecutionAdapter(ExecutionAdapter):
    """
    Upstox broker adapter.
    Uses official upstox-python-sdk.
    """
    
    def __init__(self, config_path: Path = Path("config/broker_config.json")):
        with open(config_path) as f:
            self.cfg = json.load(f)
        
        self.sandbox = self.cfg.get('sandbox', True)
        self.access_token = self.cfg.get('access_token', '')
        
        # Initialize SDK configuration
        self.configuration = upstox_client.Configuration(sandbox=self.sandbox)
        self.configuration.access_token = self.access_token
        
        self.api_client = upstox_client.ApiClient(self.configuration)
        self.order_api = upstox_client.OrderApi(self.api_client)
        self.portfolio_api = upstox_client.PortfolioApi(self.api_client)
        self.market_api = upstox_client.MarketQuoteApi(self.api_client)
        self.user_api = upstox_client.UserApi(self.api_client)
        
        self.mapper = InstrumentMapper()
        self.api_version = "2.0"
    
    def place_market_order(self, ticker: str, qty: int, side: str, tag: str = "T_Raider") -> OrderResult:
        """
        Place a delivery (CNC) market order.
        """
        instrument_token = self.mapper.get_token(ticker)
        if not instrument_token:
            return OrderResult(
                success=False,
                order_id="",
                status="REJECTED",
                filled_qty=0,
                avg_price=0.0,
                message=f"Instrument token not found for {ticker}",
                raw_response={}
            )
        
        # Upstox uses "BUY" / "SELL" (uppercase)
        transaction_type = side.upper()
        
        body = upstox_client.PlaceOrderRequest(
            quantity=qty,
            product="D",  # Delivery (CNC)
            validity="DAY",
            price=0,  # Market order
            tag=tag,
            instrument_token=instrument_token,
            order_type="MARKET",
            transaction_type=transaction_type,
            disclosed_quantity=0,
            trigger_price=0,
            is_amo=False,
            market_protection=0
        )
        
        try:
            response = self.order_api.place_order(body, self.api_version)
            data = response.data if hasattr(response, 'data') else response
            
            order_id = data.order_id if hasattr(data, 'order_id') else str(data)
            
            # Wait briefly and check status
            time.sleep(0.5)
            status = self._get_order_status(order_id)
            
            return OrderResult(
                success=status in ['complete', 'open', 'pending'],
                order_id=order_id,
                status=status.upper(),
                filled_qty=qty,  # Approximate until we fetch trades
                avg_price=0.0,
                message=f"Upstox {transaction_type} order placed",
                raw_response=data.to_dict() if hasattr(data, 'to_dict') else str(data)
            )
            
        except ApiException as e:
            return OrderResult(
                success=False,
                order_id="",
                status="REJECTED",
                filled_qty=0,
                avg_price=0.0,
                message=f"Upstox API error: {e.body}",
                raw_response={"exception": str(e)}
            )
        except Exception as e:
            return OrderResult(
                success=False,
                order_id="",
                status="REJECTED",
                filled_qty=0,
                avg_price=0.0,
                message=f"Unexpected error: {str(e)}",
                raw_response={}
            )
    
    def get_portfolio_snapshot(self, internal_holdings: Dict = None) -> PortfolioSnapshot:
        """Fetch real holdings and funds from Upstox."""
        try:
            # Get funds
            funds_resp = self.user_api.get_user_fund_margin(self.api_version)
            funds_data = funds_resp.data if hasattr(funds_resp, 'data') else funds_resp
            cash = float(funds_data.available_opening_balance) if hasattr(funds_data, 'available_opening_balance') else 0.0
            
            # Get holdings
            holdings_resp = self.portfolio_api.get_holdings(self.api_version)
            holdings_data = holdings_resp.data if hasattr(holdings_resp, 'data') else holdings_resp
            
            broker_holdings = {}
            market_value = 0.0
            
            if hasattr(holdings_data, '__iter__'):
                for h in holdings_data:
                    sym = getattr(h, 'trading_symbol', getattr(h, 'tradingsymbol', ''))
                    qty = int(getattr(h, 'quantity', 0))
                    avg = float(getattr(h, 'average_price', 0))
                    ltp = float(getattr(h, 'last_price', 0))
                    
                    broker_holdings[sym] = {"qty": qty, "avg_price": avg}
                    market_value += qty * ltp
            
            return PortfolioSnapshot(
                cash=cash,
                market_value=market_value,
                total_value=cash + market_value,
                holdings=broker_holdings
            )
            
        except Exception as e:
            # Fallback to internal state if broker fetch fails
            print(f"⚠️ Broker portfolio fetch failed: {e}")
            return PortfolioSnapshot(cash=0, market_value=0, total_value=0, holdings={})
    
    def get_ltp(self, ticker: str) -> float:
        """Get Last Traded Price via Upstox Market Quote API."""
        instrument_token = self.mapper.get_token(ticker)
        if not instrument_token:
            return 0.0
        
        try:
            resp = self.market_api.ltp(instrument_token, self.api_version)
            data = resp.data if hasattr(resp, 'data') else resp
            # Response structure varies; handle common patterns
            if hasattr(data, 'last_price'):
                return float(data.last_price)
            return 0.0
        except Exception as e:
            print(f"⚠️ LTP fetch failed for {ticker}: {e}")
            return 0.0
    
    def is_order_complete(self, order_id: str) -> bool:
        try:
            status = self._get_order_status(order_id)
            return status == 'complete'
        except Exception:
            return False
    
    def _get_order_status(self, order_id: str) -> str:
        """Fetch order status from broker."""
        try:
            resp = self.order_api.get_order_details(order_id, self.api_version)
            data = resp.data if hasattr(resp, 'data') else resp
            return getattr(data, 'status', 'unknown').lower()
        except Exception:
            return 'unknown'