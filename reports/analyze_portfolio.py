#!/usr/bin/env python3
"""
T_Raider Portfolio Analyzer
============================
Ad-hoc script to analyze and visualize a T_Raider portfolio.json trade
history: realized P&L (FIFO), win/loss stats, P&L by ticker and exit
reason, holding-period distribution, churn/fee-drag flags, and an
estimated Upstox fee drag using the current delivery fee schedule.

USAGE
-----
    python analyze_portfolio.py                       # looks for ./portfolio.json
    python analyze_portfolio.py /path/to/portfolio.json
    python analyze_portfolio.py portfolio.json -o report_out

Outputs a single PNG dashboard (default: portfolio_report.png) plus a
console summary. No internet access or extra dependencies required
beyond matplotlib (already a core project dependency).
"""

import argparse
import json
import sys
import datetime as dt
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # safe for headless / cron / CI runs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ----------------------------------------------------------------------
# Upstox delivery fee model (edit here if Upstox changes its schedule)
# ----------------------------------------------------------------------
BROKERAGE_PER_ORDER = 20.0
STT_RATE = 0.001          # 0.1% delivery, both buy & sell
EXCHANGE_RATE = 0.0000307  # NSE ~0.00307%
SEBI_RATE = 10 / 1e7       # Rs 10 / crore
STAMP_DUTY_BUY_RATE = 0.00015   # 0.015%, buy side only
DP_CHARGE = 20.0 * 1.18         # Rs 20 + 18% GST, per scrip per sell-day
GST_RATE = 0.18


def estimate_fees(side: str, value: float) -> float:
    """Rough per-fill Upstox delivery fee estimate."""
    brokerage = BROKERAGE_PER_ORDER
    stt = value * STT_RATE
    exch = value * EXCHANGE_RATE
    sebi = value * SEBI_RATE
    gst = GST_RATE * (brokerage + exch)
    stamp = value * STAMP_DUTY_BUY_RATE if side == "buy" else 0.0
    dp = DP_CHARGE if side == "sell" else 0.0
    return brokerage + stt + exch + sebi + gst + stamp + dp


# ----------------------------------------------------------------------
# Core analysis
# ----------------------------------------------------------------------
def load_history(path: Path):
    data = json.loads(path.read_text())
    return data.get("history", []), data.get("cash"), data.get("holdings", {})


def fifo_match(history):
    """
    Match buys to sells per ticker on a FIFO basis.
    Returns a list of realized round-trip dicts.
    """
    lots = defaultdict(deque)      # ticker -> deque of [qty, price, timestamp]
    trips = []

    for t in history:
        ticker, side, qty, price = t["ticker"], t["side"], t["qty"], t["price"]
        ts = t["timestamp"]

        if side == "buy":
            lots[ticker].append([qty, price, ts])
        else:
            remaining = qty
            while remaining > 0 and lots[ticker]:
                lot_qty, lot_price, lot_ts = lots[ticker][0]
                matched = min(remaining, lot_qty)
                pnl = matched * (price - lot_price)
                days_held = (
                    dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    - dt.datetime.strptime(lot_ts, "%Y-%m-%d %H:%M:%S")
                ).total_seconds() / 86400

                trips.append({
                    "ticker": ticker,
                    "qty": matched,
                    "buy_price": lot_price,
                    "sell_price": price,
                    "pnl": pnl,
                    "pnl_pct": (price - lot_price) / lot_price * 100 if lot_price else 0,
                    "strategy": t["strategy"],
                    "exit_reason": normalize_reason(t["strategy"]),
                    "sell_ts": ts,
                    "buy_ts": lot_ts,
                    "days_held": days_held,
                })

                lot_qty -= matched
                remaining -= matched
                if lot_qty == 0:
                    lots[ticker].popleft()
                else:
                    lots[ticker][0][0] = lot_qty

    return trips


def normalize_reason(strategy_str: str) -> str:
    reason = strategy_str.split("(")[0].strip()
    if "\u2192" in reason:
        reason = reason.split("\u2192")[0].strip()
    if "->" in reason:
        reason = reason.split("->")[0].strip()
    return reason


def build_equity_curve(history):
    """Approximate cumulative realized P&L over time from FIFO trips,
    ordered by sell timestamp (a simple, dependency-free equity proxy)."""
    trips = fifo_match(history)
    trips_sorted = sorted(trips, key=lambda r: r["sell_ts"])
    dates, cum_pnl = [], []
    running = 0.0
    for r in trips_sorted:
        running += r["pnl"]
        dates.append(dt.datetime.strptime(r["sell_ts"], "%Y-%m-%d %H:%M:%S"))
        cum_pnl.append(running)
    return dates, cum_pnl, trips_sorted


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def make_dashboard(history, cash, out_path: Path):
    trips = fifo_match(history)
    if not trips:
        print("No completed (matched) round-trips found — nothing to plot.")
        return

    dates, cum_pnl, trips_sorted = build_equity_curve(history)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("T_Raider Portfolio — Ad-hoc Analysis", fontsize=16, fontweight="bold")
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.28)

    # 1. Cumulative realized P&L over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, cum_pnl, color="#1f77b4", linewidth=2)
    ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax1.fill_between(dates, cum_pnl, 0, where=[v >= 0 for v in cum_pnl],
                      color="#1f77b4", alpha=0.15)
    ax1.set_title(f"Cumulative Realized P&L (FIFO)  —  Total: Rs {cum_pnl[-1]:,.0f}")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.set_ylabel("Rs")
    ax1.grid(alpha=0.3)

    # 2. P&L by ticker (top 15 abs)
    ax2 = fig.add_subplot(gs[1, 0])
    by_ticker = defaultdict(float)
    for r in trips:
        by_ticker[r["ticker"]] += r["pnl"]
    top = sorted(by_ticker.items(), key=lambda x: x[1])[-15:]
    tickers = [k.replace(".NS", "") for k, _ in top]
    vals = [v for _, v in top]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    ax2.barh(tickers, vals, color=colors)
    ax2.set_title("P&L by Ticker (top 15)")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Rs")

    # 3. P&L by exit reason
    ax3 = fig.add_subplot(gs[1, 1])
    by_reason = defaultdict(float)
    for r in trips:
        by_reason[r["exit_reason"]] += r["pnl"]
    reasons = sorted(by_reason.items(), key=lambda x: x[1])
    labels = [k for k, _ in reasons]
    vals2 = [v for _, v in reasons]
    colors2 = ["#2ca02c" if v >= 0 else "#d62728" for v in vals2]
    ax3.barh(labels, vals2, color=colors2)
    ax3.set_title("P&L by Exit Reason")
    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.set_xlabel("Rs")

    # 4. Holding period distribution
    ax4 = fig.add_subplot(gs[2, 0])
    days = [r["days_held"] for r in trips]
    ax4.hist(days, bins=min(30, max(5, len(set(days)))), color="#9467bd", edgecolor="white")
    ax4.axvline(2, color="red", linestyle="--", linewidth=1, label="2-day churn threshold")
    ax4.set_title("Holding Period Distribution (days)")
    ax4.set_xlabel("Days held")
    ax4.set_ylabel("Round-trips")
    ax4.legend()

    # 5. Win/loss + fee drag summary (text panel)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    wins = [r for r in trips if r["pnl"] > 0]
    losses = [r for r in trips if r["pnl"] <= 0]
    total_pnl = sum(r["pnl"] for r in trips)
    win_rate = len(wins) / len(trips) * 100 if trips else 0
    avg_win = sum(r["pnl"] for r in wins) / len(wins) if wins else 0
    avg_loss = sum(r["pnl"] for r in losses) / len(losses) if losses else 0
    fast_trips = [r for r in trips if r["days_held"] < 2]

    total_fees = sum(estimate_fees(t["side"], t["total"]) for t in history)

    summary = (
        f"Round-trips matched:      {len(trips)}\n"
        f"Win rate:                 {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)\n"
        f"Avg win / avg loss:       Rs {avg_win:,.0f} / Rs {avg_loss:,.0f}\n"
        f"Total realized P&L:       Rs {total_pnl:,.0f}\n"
        f"\n"
        f"Fills held < 2 days:      {len(fast_trips)} "
        f"({len(fast_trips)/len(trips)*100:.0f}% of trips)\n"
        f"P&L from <2-day trips:    Rs {sum(r['pnl'] for r in fast_trips):,.0f}\n"
        f"\n"
        f"Est. total Upstox fees:   Rs {total_fees:,.0f}\n"
        f"Fees as % of gross P&L:   {total_fees/total_pnl*100 if total_pnl else float('nan'):.0f}%\n"
        f"Est. net P&L after fees:  Rs {total_pnl - total_fees:,.0f}\n"
        f"\n"
        f"Ending cash:              Rs {cash:,.0f}" if cash is not None else ""
    )
    ax5.text(0.02, 0.98, summary, transform=ax5.transAxes, fontsize=11,
              verticalalignment="top", family="monospace",
              bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#cccccc"))
    ax5.set_title("Summary")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dashboard -> {out_path.resolve()}")

    # Console summary too, for quick CLI checks
    print("\n--- Quick Summary ---")
    print(summary)


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize a T_Raider portfolio.json")
    parser.add_argument("portfolio", nargs="?", default="portfolio.json",
                         help="Path to portfolio.json (default: ./portfolio.json)")
    parser.add_argument("-o", "--output", default="portfolio_report.png",
                         help="Output PNG path (default: portfolio_report.png)")
    args = parser.parse_args()

    path = Path(args.portfolio)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    history, cash, holdings = load_history(path)
    if not history:
        print("No trade history found in file.", file=sys.stderr)
        sys.exit(1)

    make_dashboard(history, cash, Path(args.output))


if __name__ == "__main__":
    main()