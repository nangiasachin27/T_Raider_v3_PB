import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple

class BetaFilter:
    """
    Computes individual asset Beta and tracks total weighted Portfolio Beta 
    against a shifting regulatory cap based on general index market regimes.
    """
    LOOKBACK_DAYS = 90

    @classmethod
    def calculate_stock_beta(cls, ticker: str, full_market_data: pd.DataFrame) -> float:
        """
        Calculates a stock's Beta over the last 90 days against the Nifty 50 (^NSEI).
        """
        try:
            from ingestion.data_ingestion import get_stock_data
            
            # Extract stock returns
            stock_df = get_stock_data(full_market_data, ticker)
            if stock_df.empty: return 1.0
            
            p_col = 'Adj Close' if 'Adj Close' in stock_df.columns else 'Close'
            stock_prices = stock_df[p_col].tail(cls.LOOKBACK_DAYS).dropna()
            stock_returns = stock_prices.pct_change().dropna()
            
            # Download matched index data
            index_df = yf.download("^NSEI", period="120d", progress=False)
            index_prices = index_df['Close'].squeeze().tail(cls.LOOKBACK_DAYS).dropna()
            index_returns = index_prices.pct_change().dropna()
            
            # Align return signatures
            combined = pd.concat([stock_returns, index_returns], axis=1).dropna()
            if len(combined) < 20: 
                return 1.0  # Safe default signature allocation
                
            # Beta calculation formula: Covariance(Asset, Market) / Variance(Market)
            covariance = combined.cov().iloc[0, 1]
            market_variance = combined.iloc[:, 1].var()
            
            if market_variance > 0:
                return float(covariance / market_variance)
            return 1.0
        except Exception:
            return 1.0  # Resilient fallback state

    @classmethod
    def check_portfolio_beta_gate(cls, candidate_ticker: str, portfolio: Dict, 
                                  full_market_data: pd.DataFrame, is_uptrend: bool) -> Tuple[bool, str]:
        """
        Determines if adding this asset violates our adaptive Portfolio Beta constraints.
        Enforces a maximum total Portfolio Beta constraint of 0.65 during macro downtrends.
        """
        holdings = portfolio.get("holdings", {})
        
        # Set dynamic beta boundary rule based on market trend condition
        beta_ceiling_cap = 1.50 if is_uptrend else 0.65
        
        candidate_beta = cls.calculate_stock_beta(candidate_ticker, full_market_data)
        
        if not holdings:
            if candidate_beta > beta_ceiling_cap:
                return False, f"Blocked: Single asset Beta ({candidate_beta:.2f}) exceeds current safe Cap ({beta_ceiling_cap:.2f})"
            return True, f"Approved: Asset Beta ({candidate_beta:.2f}) safe."

        # Compute current weighted portfolio beta allocation
        total_value = 0.0
        weighted_beta_sum = 0.0
        
        # Calculate for existing assets
        for held_ticker, hdata in holdings.items():
            qty = hdata['qty'] if isinstance(hdata, dict) else int(hdata or 0)
            entry_price = hdata['entry_price'] if isinstance(hdata, dict) else 1.0
            asset_value = qty * entry_price
            
            asset_beta = cls.calculate_stock_beta(held_ticker, full_market_data)
            weighted_beta_sum += (asset_beta * asset_value)
            total_value += asset_value
            
        # Hypothetically add candidate to check the frontier boundary
        # Assume standard portfolio chunk baseline allocation for weight estimation
        simulated_alloc_value = (total_value / len(holdings)) if len(holdings) > 0 else 20000.0
        
        projected_total_value = total_value + simulated_alloc_value
        projected_weighted_beta = (weighted_beta_sum + (candidate_beta * simulated_alloc_value)) / projected_total_value
        
        if projected_weighted_beta > beta_ceiling_cap:
            return False, f"SKIP: Portfolio Beta would rise to {projected_weighted_beta:.2f} exceeding Cap ceiling limit ({beta_ceiling_cap:.2f})"
            
        return True, f"Passed: Expected Portfolio Beta will be safely balanced at {projected_weighted_beta:.2f}"