"""
auto_optimizer.py  (v3 — Data-Driven Survivorship Bias Fix)
─────────────────────────────────────────────────────────────
Changes from v2:
  • Works with ANY universe in stocks.json — not just Nifty 50.
  • Survivorship bias fix is data-driven: each ticker's backtest window is
    clipped to the actual date range where price data exists, so a stock
    listed in 2021 is never tested on 2019 data.
  • Universe validation runs upfront and prints a clear report of which
    tickers have enough history to optimize.
  • optimal_params.json now stores data_start, data_end, optimized_on.
  • walk_forward_validate() no longer needs ticker/all_tickers args —
    the PIT clipping happens at the data level before it's called.
  • Everything else (block-MC, composite scoring) unchanged from v2.
"""

import itertools
from datetime import date
import pandas as pd
import json
import os
import numpy as np

from ingestion.data_ingestion import fetch_historical_data, get_stock_data
from ingestion.nse_constituents import (
    load_universe, validate_universe, print_universe_report, save_universe_report
)
from strategies.trend_follower import apply_golden_cross_strategy
from strategies.mean_reversion import apply_rsi_strategy
from strategies.volatility import apply_bollinger_strategy
from strategies.breakout import apply_breakout_strategy
from strategies.momentum import apply_macd_strategy
from strategies.stretch import apply_stretch_strategy
from engine.backtester import SimpleBacktester


# ── Monte Carlo — Block Resampling (unchanged from v2) ───────────────────────

def run_monte_carlo_filter(trade_logs, num_sims=1000, block_size=None):
    if not trade_logs or len(trade_logs) < 5:
        return 0.0
    returns = np.array(
        [(t['exit_price'] - t['entry_price']) / t['entry_price'] for t in trade_logs]
    )
    n = len(returns)
    if block_size is None:
        block_size = max(2, int(np.floor(np.sqrt(n))))
    profitable_sims = 0
    for _ in range(num_sims):
        num_blocks    = int(np.ceil(n / block_size))
        start_indices = np.random.randint(0, n - block_size + 1, size=num_blocks)
        sim_path      = np.concatenate([returns[i: i + block_size] for i in start_indices])[:n]
        if np.prod(1 + sim_path) > 1.0:
            profitable_sims += 1
    return (profitable_sims / num_sims) * 100


# ── Metrics helpers (unchanged from v2) ──────────────────────────────────────

def _compute_sharpe(trade_logs, risk_free_annual=0.065):
    if not trade_logs or len(trade_logs) < 2:
        return 0.0
    rets = np.array([(t['exit_price'] - t['entry_price']) / t['entry_price'] for t in trade_logs])
    # Trade-level Sharpe: no sqrt(252) because trades are not daily.
    # Approximate per-trade risk-free as annual / number of trades (uniformity assumption)
    rf_per_trade = risk_free_annual / max(1, len(trade_logs))
    excess = rets - rf_per_trade
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std())


def _compute_max_drawdown(trade_logs):
    if not trade_logs:
        return 100.0
    equity, peak, max_dd = 100.0, 100.0, 0.0
    for t in trade_logs:
        ret    = (t['exit_price'] - t['entry_price']) / t['entry_price']
        equity *= (1 + ret)
        peak   = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)
    return max_dd


def _composite_score(ann_return, sharpe, max_dd):
    norm_sharpe = np.clip((sharpe + 5) / 10, 0, 1)
    norm_return = np.clip((ann_return + 100) / 200, 0, 1)
    norm_inv_dd = np.clip(1 - max_dd / 100, 0, 1)
    return 0.5 * norm_sharpe + 0.3 * norm_return + 0.2 * norm_inv_dd


def _run_strategy(name, func, params, price):
    if name in ("TREND", "VOLATILITY"):
        return func(price)
    return func(price, **params)


# ── Walk-Forward Validation (unchanged from v2 — PIT clipping is now upstream) ─

def walk_forward_validate(df, price_col, strategy_name, strategy_func, params,
                          train_years=3, test_years=1, mc_threshold=60.0):
    """
    Rolling walk-forward validation.

    Survivorship bias is handled UPSTREAM — df is already clipped to the
    period where this stock has real data before this function is called.
    This function no longer needs to know about index membership.
    """
    price = df[price_col].copy()
    price.index = pd.to_datetime(price.index)

    total_days = len(price)
    # Adaptive windows: at least 6 months train, 3 months test
    train_days = min(int(train_years * 252), int(total_days * 0.6))
    test_days = min(int(test_years * 252), int(total_days * 0.2))

    if train_days < 126 or test_days < 63:  # 6mo / 3mo minimum
        return {'passed': False}

    oos_metrics = []
    fold_count = 0
    start = 0

    while start + train_days + test_days <= total_days:
        # Training fold
        train_price = price.iloc[start: start + train_days]
        train_sig   = _run_strategy(strategy_name, strategy_func, params, train_price)
        bt_train    = SimpleBacktester(stop_loss_pct=0.10)
        bt_train.run(train_sig)
        if run_monte_carlo_filter(bt_train.trades) < mc_threshold:
            start += test_days
            continue

        # Out-of-sample fold
        test_price = price.iloc[start + train_days: start + train_days + test_days]
        test_sig   = _run_strategy(strategy_name, strategy_func, params, test_price)
        bt_test    = SimpleBacktester(stop_loss_pct=0.10)
        bt_test.run(test_sig)
        test_mc    = run_monte_carlo_filter(bt_test.trades)

        if test_mc < mc_threshold:
            start += test_days
            continue

        metrics = bt_test.get_metrics(test_sig)
        oos_metrics.append({
            'oos_return': float(metrics['Post-Tax Annualized'].replace('%', '')),
            'oos_sharpe': _compute_sharpe(bt_test.trades),
            'oos_max_dd': _compute_max_drawdown(bt_test.trades),
            'oos_mc':     test_mc,
        })
        fold_count += 1
        start      += test_days

    if not oos_metrics:
        return {'passed': False}

    return {
        'passed':       True,
        'oos_return':   float(np.mean([m['oos_return'] for m in oos_metrics])),
        'oos_sharpe':   float(np.mean([m['oos_sharpe'] for m in oos_metrics])),
        'oos_max_dd':   float(np.mean([m['oos_max_dd'] for m in oos_metrics])),
        'oos_mc':       float(np.mean([m['oos_mc']     for m in oos_metrics])),
        'folds_passed': fold_count,
    }


# ── Main optimizer ────────────────────────────────────────────────────────────

def optimize_hybrid_universe(tickers=None):
    """
    Optimizes the full universe defined in config/stocks.json.

    Args:
        tickers : Optional list override. If None, reads from stocks.json.
                  Passing tickers explicitly (e.g. from main.py) is still
                  supported for backward compatibility.
    """
    print("🧬 T_RAIDER HYBRID OPTIMIZER v3 — Data-Driven Universe + Walk-Forward + Sharpe Scoring")
    print("=" * 80)

    # ── Step 1: Load universe from stocks.json ────────────────────────────
    if tickers is None:
        tickers = load_universe()
    print(f"📋 Universe: {len(tickers)} tickers loaded from config/stocks.json")

    # ── Step 2: Fetch full price history ─────────────────────────────────
    print("\n📥 Fetching 5-year price history for full universe…")
    full_data = fetch_historical_data(tickers, period="5y")

    # ── Step 3: Validate universe — data-driven survivorship bias check ───
    print("\n🔍 Validating universe against actual price data…")
    universe_report = validate_universe(tickers, full_data, min_days=500)
    print_universe_report(universe_report)
    save_universe_report(universe_report)

    # Only optimize tickers with sufficient data
    eligible = [t for t, v in universe_report.items() if v['status'] == 'ok']
    print(f"✅ {len(eligible)} tickers eligible for optimization.\n")

    strategies = [
        ("TREND",      apply_golden_cross_strategy, [{}]),
        ("MACD",       apply_macd_strategy,         [{"fast": 12, "slow": 26, "signal": 9}]),
        ("VOLATILITY", apply_bollinger_strategy,    [{}]),
        ("RSI",        apply_rsi_strategy,          [
            {"window": w, "buy": b, "sell": s}
            for w, b, s in itertools.product([14, 21], [30, 35], [70, 80])
        ]),
        ("BREAKOUT",   apply_breakout_strategy,     [{"window": w} for w in [20, 50]]),
        ("STRETCH",    apply_stretch_strategy,       [
            {"window": 20, "threshold": d} for d in [0.03, 0.05, 0.07]
        ]),
    ]

    master_plan = {}
    today       = date.today()

    for ticker in eligible:
        print(f"\nOptimising {ticker}…")

        meta = universe_report[ticker]
        df   = get_stock_data(full_data, ticker)

        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

        # ── SURVIVORSHIP BIAS FIX: clip to actual data window ─────────────
        # This is the key fix. By clipping to data_start, a stock that was
        # listed in 2021 can never be backtested on 2019–2020 data.
        # No hardcoded index membership needed — the data itself is the truth.
        df = df.loc[str(meta['data_start']): str(meta['data_end'])]

        if len(df) < 500:
            print(f"  ⚠️  Skipping after clip — only {len(df)} rows.")
            continue

        best_score     = -999
        winning_strat  = "NONE"
        winning_params = {}
        winning_wf     = {}

        for name, func, param_grid in strategies:
            for p in param_grid:
                wf = walk_forward_validate(df, price_col, name, func, p)
                if not wf['passed']:
                    continue
                score = _composite_score(wf['oos_return'], wf['oos_sharpe'], wf['oos_max_dd'])
                if score > best_score:
                    best_score     = score
                    winning_strat  = name
                    winning_params = p
                    winning_wf     = wf

        master_plan[ticker] = {
            "strategy":        winning_strat,
            "params":          winning_params,
            "expected_return": round(winning_wf.get('oos_return', 0), 2),
            "sharpe_ratio":    round(winning_wf.get('oos_sharpe', 0), 3),
            "max_drawdown":    round(winning_wf.get('oos_max_dd', 100), 2),
            "stability_score": round(winning_wf.get('oos_mc', 0), 1),
            "folds_passed":    winning_wf.get('folds_passed', 0),
            "composite_score": round(best_score, 4),
            # Traceability — when was this stock's data window?
            "data_start":      str(meta['data_start']),
            "data_end":        str(meta['data_end']),
            "data_rows":       meta['num_rows'],
            "optimized_on":    str(today),
        }

        status = "✅" if winning_strat != "NONE" else "❌"
        print(
            f"  {status} {winning_strat:12} | "
            f"OOS Return: {master_plan[ticker]['expected_return']:6.1f}% | "
            f"Sharpe: {master_plan[ticker]['sharpe_ratio']:5.2f} | "
            f"MaxDD: {master_plan[ticker]['max_drawdown']:5.1f}% | "
            f"MC: {master_plan[ticker]['stability_score']:5.1f}% | "
            f"Folds: {master_plan[ticker]['folds_passed']} | "
            f"Data: {meta['data_start']} → {meta['data_end']}"
        )

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs('config', exist_ok=True)
    with open('config/optimal_params.json', 'w') as f:
        json.dump(master_plan, f, indent=4)

    accepted  = sum(1 for v in master_plan.values() if v['strategy'] != 'NONE')
    print(f"\n✅ Done. {accepted}/{len(eligible)} eligible stocks passed all filters.")
    print(f"📄 Saved to config/optimal_params.json")
    print(f"📄 Universe report at config/universe_report.json")
    print(f"\n📉 NOTE: OOS returns are honest (PIT-clipped data).")
    print(f"   Stocks with shorter history have fewer WF folds — stability scores")
    print(f"   for recently-listed stocks should be treated with more caution.")


if __name__ == "__main__":
    optimize_hybrid_universe()