import os
import sys
import csv
import yfinance as yf
from datetime import datetime

# Pathing to find your logger and portfolio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autopilot.logger import load_portfolio


def _calculate_realised_pnl(history: list) -> float:
    """
    Computes realised P&L by matching every sell to its corresponding buy
    cost from the trade history.

    FIX 5: The original code applied STCG to gross_profit_loss, which
    included unrealised stock market value. Tax is only owed on realised
    gains — i.e., gains from positions that have actually been sold.

    Strategy:
        For each SELL entry in history, find the cost basis of those
        shares using the weighted-average entry price across prior BUYs
        for that ticker. Realised P&L = sell_proceeds - cost_basis.

    Returns:
        Total realised P&L in ₹ (can be negative).
    """
    # Build a per-ticker running cost basis from buy history
    # cost_basis[ticker] = {'total_qty': int, 'total_cost': float}
    cost_basis: dict = {}
    realised_pnl = 0.0

    for trade in history:
        ticker = trade.get('ticker')
        side   = trade.get('side')
        qty    = trade.get('qty', 0)
        price  = trade.get('price', 0.0)
        total  = trade.get('total', qty * price)

        if side == 'buy':
            if ticker not in cost_basis:
                cost_basis[ticker] = {'total_qty': 0, 'total_cost': 0.0}
            cost_basis[ticker]['total_qty']  += qty
            cost_basis[ticker]['total_cost'] += total

        elif side == 'sell' and ticker in cost_basis:
            cb = cost_basis[ticker]
            if cb['total_qty'] > 0:
                avg_cost_per_share = cb['total_cost'] / cb['total_qty']
                cost_of_sold_shares = avg_cost_per_share * qty
                sell_proceeds = total
                realised_pnl += sell_proceeds - cost_of_sold_shares

                # Reduce the remaining cost basis
                cb['total_cost'] -= cost_of_sold_shares
                cb['total_qty']  -= qty
                if cb['total_qty'] <= 0:
                    del cost_basis[ticker]

    return realised_pnl


def _csv_already_has_today(csv_file: str, today_str: str) -> bool:
    """
    Returns True if today_str already exists as a date entry in the CSV.

    FIX 6: The original code opened the CSV in append mode with no check,
    so running tracker.py twice in one day silently wrote duplicate rows.
    The performance chart would then show two data points for the same date,
    creating a zigzag artefact or wrong averages depending on the chart code.
    """
    if not os.path.isfile(csv_file):
        return False
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] == today_str:
                return True
    return False


def generate_report():
    portfolio = load_portfolio()
    cash      = portfolio['cash']
    holdings  = portfolio['holdings']

    print("\n" + "═" * 50)
    print("📈 T_RAIDER WEEKLY PERFORMANCE TRACKER")
    print("═" * 50)

    total_market_value = 0

    if not holdings:
        print("Current Status: ALL CASH")
    else:
        print(f"{'TICKER':12} | {'QTY':5} | {'ENTRY':10} | {'LIVE PRICE':12} | {'VALUE':12}")
        print("-" * 50)

        for ticker, data in holdings.items():
            # Handle both old flat format (int) and new dict format
            if isinstance(data, dict):
                qty         = data.get('qty', 0)
                entry_price = data.get('entry_price', 0)
            else:
                qty         = data
                entry_price = 0

            # Fetch the latest price
            stock         = yf.Ticker(ticker)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            value         = qty * current_price
            total_market_value += value

            entry_str = f"₹{entry_price:.1f}" if entry_price else "-"
            print(f"{ticker:12} | {qty:<5} | {entry_str:>10} | "
                  f"₹{current_price:>10.2f} | ₹{value:>10.2f}")

    # ── 1. Gross Wealth ───────────────────────────────────────────────────────
    net_worth         = cash + total_market_value
    gross_profit_loss = net_worth - 100000
    gross_pl_pct      = (gross_profit_loss / 100000) * 100

    # ── 2. Tax — applied to REALISED gains only ───────────────────────────────
    # FIX 5: Replaced `gross_profit_loss * 0.20` with realised P&L calculation.
    # The old code taxed unrealised gains too — if you're up 30% on paper but
    # haven't sold anything, you owe ₹0 in STCG today.
    # STCG rate in India is 20% (revised from 15%, effective July 2024 budget).
    history        = portfolio.get('history', [])
    realised_pnl   = _calculate_realised_pnl(history)
    estimated_stcg_tax = realised_pnl * 0.20 if realised_pnl > 0 else 0.0

    # ── 3. True Take-Home ─────────────────────────────────────────────────────
    net_take_home_profit = gross_profit_loss - estimated_stcg_tax
    net_take_home_pct    = (net_take_home_profit / 100000) * 100

    print("-" * 50)
    print(f"CASH IN HAND       : ₹{cash:,.2f}")
    print(f"STOCK MARKET VALUE : ₹{total_market_value:,.2f}")
    print(f"CURRENT NET WORTH  : ₹{net_worth:,.2f}")

    # ── 4. Daily Snapshot ─────────────────────────────────────────────────────
    # FIX 6: Check for a duplicate date before writing. If today's row is
    # already in the CSV (e.g., workflow ran twice, or tracker.py called
    # manually after the scheduled run), overwrite that row rather than
    # appending a second one.
    csv_file  = os.path.join(os.path.dirname(__file__), '..', 'config', 'daily_equity.csv')
    today_str = datetime.now().strftime('%Y-%m-%d')

    if _csv_already_has_today(csv_file, today_str):
        # Overwrite: rewrite the whole file, replacing today's existing row
        rows_to_keep = []
        if os.path.isfile(csv_file):
            with open(csv_file, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] != today_str:
                        rows_to_keep.append(row)
        rows_to_keep.append([today_str, net_worth])

        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Re-write header if it was present, else add it
            has_header = rows_to_keep and rows_to_keep[0][0] == 'Date'
            if not has_header:
                writer.writerow(['Date', 'Net_Worth'])
            writer.writerows(rows_to_keep)
        print(f"📊 Equity snapshot updated (overwrite) for {today_str}")
    else:
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Date', 'Net_Worth'])
            writer.writerow([today_str, net_worth])
        print(f"📊 Equity snapshot saved for {today_str}")

    print("═" * 50)

    color = "🟢" if gross_profit_loss >= 0 else "🔴"

    print(f"GROSS P/L          : {color} ₹{gross_profit_loss:,.2f} ({gross_pl_pct:.2f}%)")
    print(f"  (Unrealised)     : ₹{gross_profit_loss - realised_pnl:,.2f}  "
          f"← paper gain/loss, no tax owed yet")
    print(f"  (Realised)       : ₹{realised_pnl:,.2f}  ← closed trades only")
    print(f"EST. TAX (20% STCG): 🏛️  ₹{estimated_stcg_tax:,.2f}  "
          f"← on realised gains only")
    print(f"NET TAKE-HOME P/L  : 💰 ₹{net_take_home_profit:,.2f} ({net_take_home_pct:.2f}%)")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    generate_report()