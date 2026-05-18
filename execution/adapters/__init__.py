"""
Execution adapters for T_Raider.
"""

from .base import ExecutionAdapter, OrderResult, PortfolioSnapshot
from .paper_adapter import PaperExecutionAdapter

# Upstox adapter is imported lazily to avoid SDK dependency errors
# when just running paper trading mode.

__all__ = ["ExecutionAdapter", "OrderResult", "PortfolioSnapshot", "PaperExecutionAdapter"]

def get_upstox_adapter():
    """Lazy import — call this only when Upstox mode is actually needed."""
    from .upstox_adapter import UpstoxExecutionAdapter
    return UpstoxExecutionAdapter