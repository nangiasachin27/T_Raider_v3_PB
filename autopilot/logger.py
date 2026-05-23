import json
import os
from datetime import datetime

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), '..', 'config', 'portfolio.json')


def load_portfolio():
    """Reads the current state of your wallet."""
    if not os.path.exists(PORTFOLIO_FILE):
        initial = {"cash": 100000.0, "holdings": {}, "history": []}
        save_portfolio(initial)
        return initial
    with open(PORTFOLIO_FILE, 'r') as f:
        return json.load(f)


def save_portfolio(data):
    """Writes the updated state to the wallet."""
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(data, f, indent=4)


def record_transaction(ticker, side, qty, price, strategy_name):
    """
    Updates cash, holdings, and logs trade history.

    CHANGE from v1:
        Holdings are now stored as dicts instead of plain integers:
            Old: "holdings": {"RELIANCE.NS": 20}
            New: "holdings": {"RELIANCE.NS": {"qty": 20, "entry_price": 2850.0, "entry_date": "2026-05-01"}}

        entry_price is needed by bot.py to calculate gain % for partial exits.
        entry_date  is needed by bot.py for the dead money exit rule (Tier 2).

        On partial sells (qty < held qty), entry_price and entry_date are
        preserved unchanged — only qty is reduced.

        BACKWARD COMPATIBILITY: load_portfolio() handles both old (int) and
        new (dict) holding formats so existing portfolio.json files don't break.
    """
    portfolio   = load_portfolio()
    total_value = qty * price
    holdings    = portfolio.setdefault('holdings', {})

    if side == 'buy':
        portfolio['cash'] -= total_value

        if ticker in holdings:
            # Already have a position — average up the entry price and add qty
            existing = _normalise_holding(holdings[ticker])
            old_qty   = existing['qty']
            old_price = existing['entry_price']
            new_qty   = old_qty + qty
            # Weighted average entry price
            if old_price <= 0:
                avg_price = price   # treat the new buy as fresh cost basis
            else:
                avg_price = ((old_qty * old_price) + (qty * price)) / new_qty
            holdings[ticker] = {
                'qty':         new_qty,
                'entry_price': round(avg_price, 4),
                'entry_date':  existing['entry_date'],   # keep original date
            }
        else:
            holdings[ticker] = {
                'qty':         qty,
                'entry_price': price,
                'entry_date':  datetime.now().strftime('%Y-%m-%d'),
            }

    elif side == 'sell':
        portfolio['cash'] += total_value

        if ticker not in holdings:
            print(f"⚠️  Cannot sell {ticker} — not in holdings.")
            return

        existing  = _normalise_holding(holdings[ticker])
        remaining = existing['qty'] - qty

        if remaining <= 0:
            del holdings[ticker]
        else:
            holdings[ticker] = {
                'qty':         remaining,
                'entry_price': existing['entry_price'],  # unchanged
                'entry_date':  existing['entry_date'],   # unchanged
            }

    portfolio['history'].append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker':    ticker,
        'side':      side,
        'qty':       qty,
        'price':     price,
        'strategy':  strategy_name,
        'total':     round(total_value, 2),
    })

    save_portfolio(portfolio)
    print(f"📝 Logged {side.upper()} {qty} × {ticker} @ ₹{price:.2f} "
          f"(Total: ₹{total_value:,.2f})")


def _normalise_holding(holding) -> dict:
    """
    Converts old int format to new dict format transparently.
    Allows existing portfolio.json files to keep working after this update.

    Old: 20          → {'qty': 20, 'entry_price': 0.0, 'entry_date': 'unknown'}
    New: {...}       → returned as-is
    """
    if isinstance(holding, (int, float)):
        return {
            'qty':         int(holding),
            'entry_price': 0.0,       # unknown — was not stored before
            'entry_date':  'unknown',
        }
    return holding


def get_holding_qty(ticker: str) -> int:
    """Convenience helper used by bot.py — returns just the share count."""
    portfolio = load_portfolio()
    holding   = portfolio.get('holdings', {}).get(ticker)
    if holding is None:
        return 0
    return _normalise_holding(holding)['qty']


def get_holding_entry_price(ticker: str) -> float:
    """Convenience helper used by bot.py — returns the average entry price."""
    portfolio = load_portfolio()
    holding   = portfolio.get('holdings', {}).get(ticker)
    if holding is None:
        return 0.0
    return _normalise_holding(holding)['entry_price']