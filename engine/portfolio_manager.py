import json
import os


def suggest_allocation(buy_signals, current_cash=100000):
    """
    Helps you decide how much to invest in each signal to manage risk.

    buy_signals: list of dicts with keys 'ticker' and 'price'  (canonical format
                 produced by daily_screener.py and autopilot/bot.py), OR a list
                 of (ticker, price) tuples for quick manual use.

    FIX 3: The original code assumed dicts unconditionally. The __main__ self-test
    passed tuples, causing a crash:
        TypeError: tuple indices must be integers, not str
    Both callers now work correctly via _normalise_signal().
    """
    if not buy_signals:
        print("No buy signals today. Keep your cash in the bank.")
        return

    # Rule: Never put more than 20% of your total wealth into a single stock.
    # This ensures that even if one stock goes to zero, you only lose 20%.
    max_per_stock = current_cash * 0.20

    print("\n--- T_Raider Portfolio Allocation Plan ---")
    print(f"Total Available Cash: ₹{current_cash:,.2f}")
    print(f"Number of Signals: {len(buy_signals)}")
    print(f"Suggested Max Risk per Stock: ₹{max_per_stock:,.2f}\n")

    for signal in buy_signals:
        ticker, price = _normalise_signal(signal)

        # Calculate how many shares to buy using our 20% limit
        shares_to_buy     = int(max_per_stock // price)
        actual_investment = shares_to_buy * price

        print(f"📍 {ticker}:")
        print(f"   Action      : BUY {shares_to_buy} shares")
        print(f"   At Price    : ₹{price:.2f}")
        print(f"   Total Cost  : ₹{actual_investment:,.2f}")
        print("-" * 30)


def _normalise_signal(signal):
    """
    Accepts a signal in either of the two formats used across the codebase:

        Dict (canonical):  {'ticker': 'RELIANCE.NS', 'price': 2850.0, ...}
        Tuple (shorthand): ('RELIANCE.NS', 2850.0)

    Returns (ticker: str, price: float).

    Having a single normalisation helper means any future format change only
    needs to be updated here, not scattered across every caller.
    """
    if isinstance(signal, dict):
        return signal['ticker'], float(signal['price'])
    elif isinstance(signal, (list, tuple)) and len(signal) >= 2:
        return str(signal[0]), float(signal[1])
    else:
        raise TypeError(
            f"buy_signals entries must be dicts with 'ticker'/'price' keys "
            f"or (ticker, price) tuples. Got: {type(signal).__name__!r} → {signal!r}"
        )


if __name__ == "__main__":
    # FIX 3: Self-test now uses dicts — the same format produced by
    # daily_screener.py — so this file tests what production actually runs.
    # The original used tuples, which crashed immediately on signal['ticker'].
    today_signals = [
        {"ticker": "TITAN.NS",     "price": 3500.00},
        {"ticker": "ICICIBANK.NS", "price": 1100.00},
        {"ticker": "ITC.NS",       "price":  440.00},
    ]
    suggest_allocation(today_signals, current_cash=100000)

    # Tuple format still works too — useful for quick one-liners in the REPL.
    print("\n--- Tuple format (backward-compat check) ---")
    tuple_signals = [("TITAN.NS", 3500.00), ("ICICIBANK.NS", 1100.00)]
    suggest_allocation(tuple_signals, current_cash=100000)