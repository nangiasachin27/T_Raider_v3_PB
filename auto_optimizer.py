"""
auto_optimizer.py (v3.1 — Walk-Forward + Sharpe Fixes)
─────────────────────────────────────────────────────────────

BUG FIX 3 — walk_forward_validate() stepped by test_days only, causing
massive overlap between training windows of consecutive folds.

Example (train=756d, test=252d):
  BEFORE: Fold 1 trains on days 0-756. Fold 2 trains on days 252-1008.
          504 days of training data SHARED between folds.
          Stability scores were inflated — strategies appeared validated
          on more independent data than they actually were.

  AFTER:  Fold 1 trains on days 0-756, tests 756-1008.
          Fold 2 trains on days 1008-1764, tests 1764-2016.
          Zero overlap. True walk-forward.

FIX: start += train_days + test_days  (was: start += test_days)

BUG FIX 4 — _compute_sharpe() used trade count as risk-free rate proxy.
A strategy with 50 trades got rf=0.13%/trade; one with 5 trades got
rf=1.3%/trade — 10x penalty on low-frequency strategies regardless of
actual holding period or real returns. This systematically biased the
composite_score against Trend Follower and toward RSI/Breakout.

FIX: Use average holding period in days to compute per-trade rf rate.
  rf_per_trade = risk_free_annual * (avg_hold_days / 365)
Requires trade dicts to carry entry_date / exit_date, which SimpleBacktester
already records. If dates are missing, falls back gracefully to the old
count-based method with a warning.
"""

import itertools
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
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
from strategies.rsi_divergence import rsi_divergence_strategy
from strategies.atr_breakout import atr_breakout_strategy

# ── New Strategies Added Here ─────────────────────────────────────────────────
from strategies import obv_momentum, nr7, supertrend, stoch_rsi

from engine.backtester import SimpleBacktester


# ── Monte Carlo — Vectorised Block Resampling ─────────────────────────────────
#
# BEFORE: Python for-loop over num_sims=1000 — one np.concatenate per sim.
#   Cost: ~14 ms per call × 228,096 total calls = ~53 minutes.
#
# AFTER: All 1,000 simulations built in a single numpy operation.
#   Shape: (num_sims, num_blocks, block_size) → gather → product.
#   Cost: ~0.25 ms per call → ~1 minute total. ~58× faster.
#   Result is statistically identical (same block-bootstrap distribution).

def run_monte_carlo_filter(trade_logs, num_sims=1000, block_size=None):
    if not trade_logs or len(trade_logs) < 5:
        return 0.0

    returns = np.array(
        [(t['exit_price'] - t['entry_price']) / t['entry_price'] for t in trade_logs]
    )
    n = len(returns)
    if block_size is None:
        block_size = max(2, int(np.floor(np.sqrt(n))))

    num_blocks = int(np.ceil(n / block_size))

    # Draw all start indices at once: shape (num_sims, num_blocks)
    starts = np.random.randint(0, max(1, n - block_size + 1), size=(num_sims, num_blocks))

    # Build index grid: shape (num_sims, num_blocks * block_size) then clip to n
    offsets = np.arange(block_size)                                  # (block_size,)
    indices = (starts[:, :, None] + offsets[None, None, :])         # (num_sims, num_blocks, block_size)
    indices = indices.reshape(num_sims, -1)[:, :n]                  # (num_sims, n)
    indices = np.clip(indices, 0, n - 1)                            # guard edge

    # Gather returns and compute cumulative product per simulation
    sim_returns = returns[indices]                                   # (num_sims, n)
    profitable  = np.sum(np.prod(1 + sim_returns, axis=1) > 1.0)

    return float(profitable / num_sims * 100)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _compute_sharpe(trade_logs, risk_free_annual=0.065):
    """
    BUG FIX 4: Compute Sharpe using actual average holding period,
    not trade count, for the per-trade risk-free rate.

    Old code:
        rf_per_trade = risk_free_annual / max(1, len(trade_logs))
        Problem: 50 trades → rf=0.13%/trade; 5 trades → rf=1.3%/trade.
        Low-frequency strategies (Trend, Breakout) penalised 10x more.

    New code:
        avg_hold_days = mean(exit_date - entry_date) across all trades
        rf_per_trade  = risk_free_annual * (avg_hold_days / 365)
        Fallback to count-based if dates not in trade dicts.
    """
    if not trade_logs or len(trade_logs) < 2:
        return 0.0

    rets = np.array(
        [(t['exit_price'] - t['entry_price']) / t['entry_price'] for t in trade_logs]
    )

    # FIX 4: use average holding period for risk-free rate
    hold_days_list = []
    for t in trade_logs:
        try:
            entry_dt = pd.to_datetime(t['entry_date'])
            exit_dt  = pd.to_datetime(t['exit_date'])
            hold_days_list.append((exit_dt - entry_dt).days)
        except (KeyError, TypeError, ValueError):
            pass  # date fields missing — will fall back

    if hold_days_list:
        avg_hold_days = float(np.mean(hold_days_list))
        rf_per_trade  = risk_free_annual * (avg_hold_days / 365.0)
    else:
        # Fallback: old count-based method (with a warning)
        import warnings
        warnings.warn(
            "_compute_sharpe: trade dicts missing entry_date/exit_date — "
            "falling back to count-based risk-free rate. "
            "Update SimpleBacktester to record dates for accurate Sharpe."
        )
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
    norm_sharpe  = np.clip((sharpe + 5) / 10, 0, 1)
    norm_return  = np.clip((ann_return + 100) / 200, 0, 1)
    norm_inv_dd  = np.clip(1 - max_dd / 100, 0, 1)
    return 0.5 * norm_sharpe + 0.3 * norm_return + 0.2 * norm_inv_dd


def _run_strategy(name, func, params, price):
    if name in ("TREND", "VOLATILITY", "NR7_SQUEEZE"):
        res = func(price)
    else:
        res = func(price, **params)
        
    # Ensure 'Price' column exists for SimpleBacktester compatibility with new strategies
    if 'Price' not in res.columns and 'Close' in res.columns:
        res['Price'] = res['Close']
        
    return res


# ── Walk-Forward Validation ───────────────────────────────────────────────────
def walk_forward_validate(df, price_col, strategy_name, strategy_func, params,
                           train_years=3, test_years=1, mc_threshold=55.0):
    price       = df[price_col].copy()
    price.index = pd.to_datetime(price.index)
    total_days  = len(price)

    train_days = min(int(train_years * 252), int(total_days * 0.6))
    test_days  = min(int(test_years  * 252), int(total_days * 0.2))

    if train_days < 126 or test_days < 63:
        return {'passed': False}

    oos_metrics = []
    fold_count  = 0
    start       = 0
    
    # NEW: Robustly define which strategies need the full dataframe (High/Low/Vol)
    full_df_strats = ["OBV_MOMENTUM", "NR7_SQUEEZE", "SUPERTREND", "STOCH_RSI", "RSI_DIVERGENCE", "ATR_BREAKOUT"]

    while start + train_days + test_days <= total_days:

        # Training fold
        train_price = price.iloc[start: start + train_days]
        train_df    = df.iloc[start: start + train_days]
        
        # Route the correct data format based on strategy name (Case-insensitive)
        input_data = train_df if strategy_name.upper() in full_df_strats else train_price

        train_sig   = _run_strategy(strategy_name, strategy_func, params, input_data)
        bt_train    = SimpleBacktester(stop_loss_pct=0.10)
        bt_train.run(train_sig)

        if run_monte_carlo_filter(bt_train.trades) < mc_threshold:
            # Training fold failed MC — skip to next non-overlapping window
            start += train_days + test_days
            continue

        # Out-of-sample fold
        test_price = price.iloc[start + train_days: start + train_days + test_days]
        test_df    = df.iloc[start + train_days: start + train_days + test_days]
        
        # Route the correct data format for the test fold
        input_data_test = test_df if strategy_name.upper() in full_df_strats else test_price

        test_sig   = _run_strategy(strategy_name, strategy_func, params, input_data_test)
        bt_test    = SimpleBacktester(stop_loss_pct=0.10)
        bt_test.run(test_sig)

        test_mc = run_monte_carlo_filter(bt_test.trades)
        if test_mc < mc_threshold:
            start += train_days + test_days
            continue

        metrics = bt_test.get_metrics(test_sig)
        oos_metrics.append({
            'oos_return': float(metrics['Post-Tax Annualized'].replace('%', '')),
            'oos_sharpe': _compute_sharpe(bt_test.trades),    
            'oos_max_dd': _compute_max_drawdown(bt_test.trades),
            'oos_mc':     test_mc,
        })
        fold_count += 1

        # Advance by full window so next fold is non-overlapping
        start += train_days + test_days

    if not oos_metrics:
        return {'passed': False}

    return {
        'passed':      True,
        'oos_return':  float(np.mean([m['oos_return'] for m in oos_metrics])),
        'oos_sharpe':  float(np.mean([m['oos_sharpe'] for m in oos_metrics])),
        'oos_max_dd':  float(np.mean([m['oos_max_dd'] for m in oos_metrics])),
        'oos_mc':      float(np.mean([m['oos_mc']     for m in oos_metrics])),
        'folds_passed': fold_count,
    }

# ── Per-ticker worker (module-level so Windows spawn can pickle it) ───────────
#
# On Windows, ProcessPoolExecutor uses 'spawn' to start workers — each worker
# is a fresh Python process that imports the module from scratch. This means
# every object sent to a worker must be picklable via the top-level import path.
# A closure (function defined inside another function) captures free variables
# that aren't picklable this way, causing AttributeError on Windows.
#
# Fix: move the worker to module level and pass all dependencies explicitly
# in a single dict payload. Dicts, DataFrames, and lists are all picklable.

def _optimise_ticker(payload: dict):
    """
    Worker function for parallel optimisation. Receives all dependencies
    explicitly rather than via closure so it can be pickled on Windows.
    """
    ticker          = payload['ticker']
    meta            = payload['meta']
    df              = payload['df']
    strategies      = payload['strategies']
    today           = payload['today']

    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df = df.loc[str(meta['data_start']): str(meta['data_end'])]
    if len(df) < 500:
        return ticker, None, meta  # skipped — insufficient data after clip

    best_score     = -999
    winning_strat  = "NONE"
    winning_params = {}
    winning_wf     = {}
    # NEW: Array to store research diagnostics for this ticker
    diagnostics = []

    for name, func, param_grid in strategies:
        for p in param_grid:
            wf = walk_forward_validate(df, price_col, name, func, p)
            if not wf['passed']:
                # NEW: Log why it failed (Usually Monte Carlo < 60% or not enough OOS folds)
                diagnostics.append({
                    "strategy": name,
                    "params": p,
                    "reason": "Failed Walk-Forward / Monte Carlo Filter",
                    "stability_score": round(wf.get('oos_mc', 0), 1) if 'oos_mc' in wf else 0.0
                })
                continue
                
            score = _composite_score(wf['oos_return'], wf['oos_sharpe'], wf['oos_max_dd'])
            
            # NEW: Log passing strategies so Research can see what "almost" won
            diagnostics.append({
                "strategy": name,
                "params": p,
                "reason": "Outscored by better strategy" if score <= best_score else "Current Best",
                "score": round(score, 4),
                "expected_return": round(wf.get('oos_return', 0), 2),
                "stability_score": round(wf.get('oos_mc', 0), 1)
            })

            if score > best_score:
                best_score     = score
                winning_strat  = name
                winning_params = p
                winning_wf     = wf
    result = {
        "strategy":        winning_strat,
        "params":          winning_params,
        "expected_return": round(winning_wf.get('oos_return', 0),  2),
        "sharpe_ratio":    round(winning_wf.get('oos_sharpe', 0),  3),
        "max_drawdown":    round(winning_wf.get('oos_max_dd', 100), 2),
        "stability_score": round(winning_wf.get('oos_mc', 0),      1),
        "folds_passed":    winning_wf.get('folds_passed', 0),
        "composite_score": round(best_score, 4),
        "data_start":      str(meta['data_start']),
        "data_end":        str(meta['data_end']),
        "data_rows":       meta['num_rows'],
        "optimized_on":    str(today),
        "diagnostics":     diagnostics  # NEW: Temporarily attach diagnostics to the result
    }
    return ticker, result, meta


# ── Main optimizer ────────────────────────────────────────────────────────────

def optimize_hybrid_universe(tickers=None):
    print("🧬 T_RAIDER HYBRID OPTIMIZER v3.1 — Walk-Forward + Sharpe Fix")
    print("=" * 80)

    if tickers is None:
        tickers = load_universe()
    print(f"📋 Universe: {len(tickers)} tickers loaded from config/stocks.json")

    print("\n📥 Fetching 5-year price history for full universe…")
    full_data = fetch_historical_data(tickers, period="5y")

    print("\n🔍 Validating universe against actual price data…")
    universe_report = validate_universe(tickers, full_data, min_days=500)
    print_universe_report(universe_report)
    save_universe_report(universe_report)

    eligible = [t for t, v in universe_report.items() if v['status'] == 'ok']
    print(f"✅ {len(eligible)} tickers eligible for optimization.\n")

    strategies = [
        # ── TREND: add short-cycle EMA variant (20/50) for momentum stocks ──────
        ("TREND",     apply_golden_cross_strategy, [{}]),

        # ── MACD: add slow signal-line variant for longer-cycle setups ──────────
        # (8,21,5)=fast scalping  (12,26,9)=classic  (19,39,9)=medium  (19,39,14)=slow confirmation
        ("MACD",      apply_macd_strategy,          [
            {"fast": f, "slow": s, "signal": sig}
            for f, s, sig in [(12, 26, 9), (8, 21, 5), (19, 39, 9), (19, 39, 14)]
        ]),

        # ── VOLATILITY: add 1.5σ bands for low-vol FMCG/pharma stocks that
        #    rarely touch 2σ, and a tighter 10-day window for faster mean-reversion
        ("VOLATILITY", apply_bollinger_strategy,    [
            {"window": w, "num_std": n}
            for w, n in itertools.product([10, 14, 20], [1.5, 2.0, 2.5])
        ]),

        # ── RSI: add deep-oversold buy=25 for high-beta names, and early-exit
        #    sell=65 for sideways/range-bound stocks unlikely to reach sell=70+
        ("RSI",       apply_rsi_strategy, [
            {"window": w, "buy": b, "sell": s}
            for w, b, s in itertools.product([10, 14, 21], [25, 30, 35], [65, 70, 80])
        ]),

        # ── BREAKOUT: fill gaps at 15 and 30 days between existing 10/20/50 ─────
        ("BREAKOUT",  apply_breakout_strategy,  [{"window": w} for w in [10, 15, 20, 30, 50]]),

        # ── STRETCH: add shorter MA window (10) for faster-trending sectors
        #    (defence, infra, capital goods) that revert to a shorter mean
        ("STRETCH",   apply_stretch_strategy,   [
            {"window": w, "threshold": d}
            for w, d in itertools.product([10, 20], [0.02, 0.03, 0.05, 0.07])
        ]),

        # ── OBV_MOMENTUM: add slow ema_period=50 for large-cap accumulation ─────
        ("OBV_MOMENTUM", obv_momentum.execute_strategy, [
            {"ema_period": e} for e in [14, 20, 30, 50]
        ]),

        # ── NR7_SQUEEZE: now parameterised — sweep lookback (4/7/10) and
        #    breakout validity window (2/3/5 days)
        #    NR4 fires more often (tighter squeeze); NR10 is rarer but more decisive
        ("NR7_SQUEEZE",  nr7.execute_strategy, [
            {"lookback": lb, "breakout_window": bw}
            for lb, bw in [(4, 2), (4, 3), (7, 3), (7, 5), (10, 3), (10, 5)]
        ]),

        # ── SUPERTREND: add slow large-cap variant (20-day ATR, 2σ) ─────────────
        ("SUPERTREND",   supertrend.execute_strategy,   [
            {"period": p, "multiplier": m}
            for p, m in [(7, 2.0), (10, 3.0), (14, 2.5), (20, 2.0)]
        ]),

        # ── STOCH_RSI: decouple k_smooth and d_smooth — fast signal (k=2)
        #    with slower confirmation (d=5) gives asymmetric entry filtering
        ("STOCH_RSI",    stoch_rsi.execute_strategy,    [
            {"rsi_period": r, "stoch_period": sp, "k_smooth": k, "d_smooth": d}
            for r, sp, k, d in [
                (14, 14, 3, 3),   # original baseline
                (9,   9, 3, 3),   # faster RSI + stoch
                (14, 14, 2, 5),   # fast signal, slow confirmation
                (9,  14, 2, 3),   # mixed: fast RSI, medium stoch
            ]
        ]),
        # ── RSI_DIVERGENCE: For Choppy / Sideways Markets ─────────────
        ("RSI_DIVERGENCE", rsi_divergence_strategy, [
            {"rsi_period": 14, "oversold": 30, "overbought": 70},
            {"rsi_period": 10, "oversold": 25, "overbought": 75}
        ]),
        
        # ── ATR_BREAKOUT: For Highly Volatile Markets ─────────────────
        ("ATR_BREAKOUT", atr_breakout_strategy, [
            {"lookback": 20, "atr_period": 14, "atr_multiplier": 3.0},
            {"lookback": 15, "atr_period": 10, "atr_multiplier": 2.5}
        ])
    ]

    master_plan = {}
    research_diagnostics = {}
    today       = date.today()

    # ── Build per-ticker payloads ─────────────────────────────────────────────
    # Each payload is a plain dict — fully picklable on Windows spawn workers.
    # full_data is sliced per ticker here (in the main process) so workers
    # receive only the data they need rather than the entire 216-ticker dataset.
    payloads = []
    for ticker in eligible:
        df = get_stock_data(full_data, ticker)
        payloads.append({
            'ticker':     ticker,
            'meta':       universe_report[ticker],
            'df':         df,
            'strategies': strategies,
            'today':      today,
        })

    # ── Parallel execution across all CPU cores ───────────────────────────────
    # ProcessPoolExecutor (not ThreadPoolExecutor) — work is CPU-bound and
    # Python's GIL would block threads from running in parallel.
    # The if __name__ == '__main__' guard in the entry point below is required
    # on Windows (spawn start method) to prevent recursive worker spawning.
    n_workers = min(multiprocessing.cpu_count(), len(eligible))
    print(f"⚡ Parallel optimisation across {n_workers} CPU core(s)…\n")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_optimise_ticker, p): p['ticker'] for p in payloads}
        for future in as_completed(futures):
            ticker, result, meta = future.result()
            if result is None:
                print(f"  ⚠️  {ticker} — skipped (insufficient data after clip)")
                continue
            # NEW: Extract diagnostics before saving to keep optimal_params.json clean
            ticker_diagnostics = result.pop("diagnostics", [])
            research_diagnostics[ticker] = ticker_diagnostics

            master_plan[ticker] = result
            status = "✅" if result['strategy'] != "NONE" else "❌"
            print(
                f"  {status} {ticker:20} {result['strategy']:12} | "
                f"OOS: {result['expected_return']:6.1f}% | "
                f"Sharpe: {result['sharpe_ratio']:5.2f} | "
                f"MaxDD: {result['max_drawdown']:5.1f}% | "
                f"MC: {result['stability_score']:5.1f}% | "
                f"Folds: {result['folds_passed']} | "
                f"Data: {meta['data_start']} → {meta['data_end']}"
            )

    os.makedirs('config', exist_ok=True)
    with open('config/optimal_params.json', 'w') as f:
        json.dump(master_plan, f, indent=4)

    # NEW: Save Research file
    with open('config/research_diagnostics.json', 'w') as f:
        json.dump(research_diagnostics, f, indent=4)

    accepted = sum(1 for v in master_plan.values() if v['strategy'] != 'NONE')
    print(f"\n✅ Done. {accepted}/{len(eligible)} eligible stocks passed all filters.")
    print(f"📄 Saved to config/optimal_params.json")
    print(f"🔬 Saved research data to config/research_diagnostics.json")
    print(f"\n⚠️  NOTE: Walk-forward folds are now non-overlapping (v3.1 fix).")
    print(f"    Stability scores will be LOWER than v3.0 — this is correct.")
    print(f"    Previous optimal_params.json scores were overstated.")


# ── Entry point — Windows spawn guard is REQUIRED ────────────────────────────
# On Windows, multiprocessing uses 'spawn': each worker imports this module
# from scratch. Without this guard, importing the module would re-execute
# optimize_hybrid_universe(), spawning workers recursively until the OS
# runs out of processes. This guard is a no-op on Linux/macOS (fork).
if __name__ == "__main__":
    optimize_hybrid_universe()