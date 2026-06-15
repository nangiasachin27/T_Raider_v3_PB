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

# ── New Strategies Added Here ─────────────────────────────────────────────────
from strategies import obv_momentum, nr7, supertrend, stoch_rsi

from engine.backtester import SimpleBacktester


# ── Monte Carlo — Block Resampling (unchanged) ────────────────────────────────

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
        num_blocks   = int(np.ceil(n / block_size))
        start_indices = np.random.randint(0, n - block_size + 1, size=num_blocks)
        sim_path     = np.concatenate([returns[i: i + block_size] for i in start_indices])[:n]
        if np.prod(1 + sim_path) > 1.0:
            profitable_sims += 1
    return (profitable_sims / num_sims) * 100


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
                           train_years=3, test_years=1, mc_threshold=60.0):
    """
    BUG FIX 3: Non-overlapping walk-forward folds.

    BEFORE (broken):
        start += test_days
        → Fold 2 training overlapped 75% with Fold 1 training.
        → Stability scores were inflated across ALL strategies.
        → optimal_params.json stability figures were overconfident.

    AFTER (fixed):
        start += train_days + test_days
        → Each fold is completely independent of the previous.
        → Fewer folds per stock (honest), but each fold is truly OOS.
        → Stability scores now reflect genuine out-of-sample performance.

    NOTE: After applying this fix, re-run auto_optimizer.py to rebuild
    optimal_params.json. Existing stability scores are overstated.
    """
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

    while start + train_days + test_days <= total_days:

        # Training fold
        train_price = price.iloc[start: start + train_days]
        train_df    = df.iloc[start: start + train_days]
        
        # New strategies require the full dataframe, older ones require just the price series
        input_data = train_df if strategy_name in ["OBV_MOMENTUM", "NR7_SQUEEZE", "SUPERTREND", "STOCH_RSI"] else train_price

        train_sig   = _run_strategy(strategy_name, strategy_func, params, input_data)
        bt_train    = SimpleBacktester(stop_loss_pct=0.10)
        bt_train.run(train_sig)

        if run_monte_carlo_filter(bt_train.trades) < mc_threshold:
            # Training fold failed MC — skip to next non-overlapping window
            start += train_days + test_days   # FIX 3
            continue

        # Out-of-sample fold
        test_price = price.iloc[start + train_days: start + train_days + test_days]
        test_df    = df.iloc[start + train_days: start + train_days + test_days]
        
        input_data_test = test_df if strategy_name in ["OBV_MOMENTUM", "NR7_SQUEEZE", "SUPERTREND", "STOCH_RSI"] else test_price

        test_sig   = _run_strategy(strategy_name, strategy_func, params, input_data_test)
        bt_test    = SimpleBacktester(stop_loss_pct=0.10)
        bt_test.run(test_sig)

        test_mc = run_monte_carlo_filter(bt_test.trades)
        if test_mc < mc_threshold:
            start += train_days + test_days   # FIX 3
            continue

        metrics = bt_test.get_metrics(test_sig)
        oos_metrics.append({
            'oos_return': float(metrics['Post-Tax Annualized'].replace('%', '')),
            'oos_sharpe': _compute_sharpe(bt_test.trades),    # FIX 4 applied here
            'oos_max_dd': _compute_max_drawdown(bt_test.trades),
            'oos_mc':     test_mc,
        })
        fold_count += 1

        # FIX 3: advance by full window so next fold is non-overlapping
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
        ("TREND",     apply_golden_cross_strategy, [{}]),
        ("MACD",      apply_macd_strategy,          [{"fast": 12, "slow": 26, "signal": 9}]),
        ("VOLATILITY", apply_bollinger_strategy,    [{}]),
        ("RSI",       apply_rsi_strategy, [
            {"window": w, "buy": b, "sell": s}
            for w, b, s in itertools.product([14, 21], [30, 35], [70, 80])
        ]),
        ("BREAKOUT",  apply_breakout_strategy,  [{"window": w} for w in [20, 50]]),
        ("STRETCH",   apply_stretch_strategy,   [
            {"window": 20, "threshold": d} for d in [0.03, 0.05, 0.07]
        ]),
        # ── 4 New Strategies Added Below ──────────────────────────────────────
        ("OBV_MOMENTUM", obv_momentum.execute_strategy, [{"ema_period": 20}, {"ema_period": 14}]),
        ("NR7_SQUEEZE",  nr7.execute_strategy,          [{}]),
        ("SUPERTREND",   supertrend.execute_strategy,   [{"period": 10, "multiplier": 3.0}, {"period": 14, "multiplier": 2.5}]),
        ("STOCH_RSI",    stoch_rsi.execute_strategy,    [{"rsi_period": 14, "stoch_period": 14, "k_smooth": 3, "d_smooth": 3}]),
    ]

    master_plan = {}
    today       = date.today()

    for ticker in eligible:
        print(f"\nOptimising {ticker}…")
        meta     = universe_report[ticker]
        df       = get_stock_data(full_data, ticker)
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

        # Survivorship bias fix: clip to actual data window
        df = df.loc[str(meta['data_start']): str(meta['data_end'])]
        if len(df) < 500:
            print(f"  ⚠️ Skipping after clip — only {len(df)} rows.")
            continue

        best_score    = -999
        winning_strat = "NONE"
        winning_params = {}
        winning_wf    = {}

        for name, func, param_grid in strategies:
            for p in param_grid:
                # FIX 3 + FIX 4 applied inside walk_forward_validate
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

    os.makedirs('config', exist_ok=True)
    with open('config/optimal_params.json', 'w') as f:
        json.dump(master_plan, f, indent=4)

    accepted = sum(1 for v in master_plan.values() if v['strategy'] != 'NONE')
    print(f"\n✅ Done. {accepted}/{len(eligible)} eligible stocks passed all filters.")
    print(f"📄 Saved to config/optimal_params.json")
    print(f"\n⚠️  NOTE: Walk-forward folds are now non-overlapping (v3.1 fix).")
    print(f"    Stability scores will be LOWER than v3.0 — this is correct.")
    print(f"    Previous optimal_params.json scores were overstated.")


if __name__ == "__main__":
    optimize_hybrid_universe()