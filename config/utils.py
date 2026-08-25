import json
from pathlib import Path
from functools import lru_cache

def get_config_tickers(key="nifty_50"):
    """Loads the ticker list from config/stocks.json"""
    with open('config/stocks.json', 'r') as f:
        config = json.load(f)
    return config.get(key, [])


@lru_cache(maxsize=1)
def get_fee_config() -> dict:
    """
    Loads config/fee_config.json — single source of truth for Upstox fee
    rates and fee-aware sizing/rebalance thresholds. Cached for the life
    of the process; delete the process (or call get_fee_config.cache_clear())
    to pick up edits mid-run.
    """
    path = Path("config/fee_config.json")
    if not path.exists():
        raise FileNotFoundError(
            "config/fee_config.json not found — required for fee-aware sizing."
        )
    with open(path) as f:
        return json.load(f)


def estimate_round_trip_fee(notional: float) -> float:
    """
    Rough Upstox equity-delivery round-trip fee estimate (buy + sell) for a
    given trade notional, using rates from config/fee_config.json.
    """
    cfg = get_fee_config()
    brokerage = cfg["brokerage_per_order_inr"]
    dp = cfg["dp_charge_inr"]
    stt = notional * cfg["stt_rate"]
    exch = notional * cfg["exchange_txn_rate"] * 2  # both legs
    sebi = notional * cfg["sebi_rate"] * 2
    stamp = notional * cfg["stamp_duty_buy_rate"]
    gst = cfg["gst_rate"] * (brokerage * 2 + notional * cfg["exchange_txn_rate"] * 2)
    return (brokerage * 2) + dp + stt + exch + sebi + stamp + gst


def is_fee_efficient(notional: float, max_fee_pct: float = None) -> bool:
    """
    Returns True if the estimated round-trip fee stays under the configured
    max_fee_pct_of_notional threshold for this trade size.
    """
    if notional <= 0:
        return False
    cfg = get_fee_config()
    threshold = max_fee_pct if max_fee_pct is not None else cfg["sizing"]["max_fee_pct_of_notional"]
    return (estimate_round_trip_fee(notional) / notional) <= threshold