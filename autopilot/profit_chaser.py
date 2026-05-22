"""
autopilot/profit_chaser.py  (v1.1)
────────────────────────────────────────────────────────────────────────────────
Active Profit Chaser — bridges the gap between current portfolio return and the
quarterly_config.json target by rotating weak→strong holdings.

Fixes in v1.1:
    FIX 1: Removed dead code after return in portfolio_current_return().
    FIX 2: Replaced cfg.setdefault("chaser",{}) with cfg["chaser"] at save call.
    FIX 3: save_quarterly_config() now raises if chaser block is missing.
    FIX 4: portfolio_current_return() mirrors tracker.py exactly.
    FIX 5: last_chaser_run stamped on ALL early-exit paths, not just happy path.

Usage:
    python -m autopilot.profit_chaser --mode CONSERVATIVE          # dry run
    python -m autopilot.profit_chaser --mode BALANCED              # dry run
    python -m autopilot.profit_chaser --mode CONSERVATIVE --execute
"""

from __future__ import annotations

import json
import argparse
from datetime import date, datetime
from pathlib import Path
import os
import sys

import pandas as pd
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autopilot.logger import load_portfolio
from ingestion.nse_constituents import get_market_regime
from daily_screener import calculate_atr, get_nifty_drop_from_50d_high
from macro_filter import MacroFilter, MARKET_CONFIGS, FilterAction

QUARTERLY_CONFIG_PATH = Path("config/quarterly_config.json")
OPTIMAL_PARAMS_PATH   = Path("config/optimal_params.json")
CHASER_LOG_PATH       = Path("config/manual_logs/chaser_log.json")


# ════════════════════════════════════════════════════════════════════════════
# 1. Config helpers
# ════════════════════════════════════════════════════════════════════════════

def load_quarterly_config() -> dict:
    with open(QUARTERLY_CONFIG_PATH) as f:
        return json.load(f)


def save_quarterly_config(cfg: dict) -> None:
    # FIX 3: Safety guard — raises loudly instead of silently nuking chaser block.
    if "chaser" not in cfg:
        raise ValueError(
            "save_quarterly_config: 'chaser' block missing — "
            "aborting save to prevent data loss."
        )
    with open(QUARTERLY_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print("  💾 quarterly_config.json saved.")


def load_optimal_params() -> dict:
    if not OPTIMAL_PARAMS_PATH.exists():
        raise FileNotFoundError(
            "config/optimal_params.json not found. Run auto_optimizer.py first."
        )
    with open(OPTIMAL_PARAMS_PATH) as f:
        return json.load(f)


def _chaser_cfg(cfg: dict) -> dict:
    """
    Ensures cfg["chaser"] exists and returns a LIVE REFERENCE (not a copy).
    Any mutation to the returned dict is reflected in cfg, so
    save_quarterly_config(cfg) always captures chaser updates correctly.
    """
    if "chaser" not in cfg:
        cfg["chaser"] = {}
    return cfg["chaser"]


def _mode_overrides(chaser: dict, mode: str) -> dict:
    """Merge base chaser defaults with the selected mode's overrides."""
    base = {
        "weak_threshold":         chaser.get("weak_threshold",   -0.05),
        "strong_min_score":       chaser.get("strong_min_score",  0.70),
        "buy_in_downtrend":       False,
        "nifty_dip_pct_required": 0.05,
    }
    overrides = chaser.get("mode_overrides", {}).get(mode, {})
    return {**base, **overrides}


# ════════════════════════════════════════════════════════════════════════════
# 2. Quarter maths
# ════════════════════════════════════════════════════════════════════════════

def _quarter_end(cfg: dict) -> date:
    start = date.fromisoformat(cfg["quarter_start_date"])
    return date(
        start.year + (start.month + cfg["quarter_days"] // 30 - 1) // 12,
        (start.month + cfg["quarter_days"] // 30 - 1) % 12 + 1,
        start.day,
    )


def days_left(cfg: dict) -> int:
    return max((_quarter_end(cfg) - date.today()).days, 0)


def urgency_multiplier(cfg: dict, chaser: dict) -> float:
    """
    Scales aggressiveness as quarter deadline approaches.
    1.0 (normal) -> 1.5 (moderate) -> 2.0 (last stretch).
    """
    urg = chaser.get("urgency_scaling", {})
    if not urg.get("enabled", True):
        return 1.0
    dl = days_left(cfg)
    if dl <= urg.get("days_left_aggressive", 20):
        return urg.get("multiplier_aggressive", 2.0)
    if dl <= urg.get("days_left_moderate", 45):
        return urg.get("multiplier_moderate", 1.5)
    return 1.0


# ════════════════════════════════════════════════════════════════════════════
# 3. Portfolio return & position scoring
# ════════════════════════════════════════════════════════════════════════════

def fetch_latest_prices(tickers: list[str]) -> pd.Series:
    """Batch-download latest close prices for all tickers."""
    if not tickers:
        return pd.Series(dtype=float)
    raw = yf.download(tickers, period="5d", progress=False, auto_adjust=True)
    if raw.empty:
        return pd.Series(dtype=float)
    close = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(close, pd.DataFrame):
        return close.iloc[-1]
    return close


def portfolio_current_return(portfolio: dict, prices: pd.Series, cfg: dict) -> float:
    """
    FIX 4: Mirrors tracker.py exactly.

    tracker.py:
        net_worth         = cash + total_market_value
        gross_profit_loss = net_worth - 100000          (hardcoded original capital)
        gross_pl_pct      = gross_profit_loss / 100000

    Previous broken version used sum(qty x entry_price) as denominator and
    excluded cash, causing ~4x understatement vs tracker output.
    """
    holdings = portfolio.get("holdings", {})
    cash     = float(portfolio.get("cash", 0))

    total_market_value = 0.0
    for ticker, data in holdings.items():
        if isinstance(data, dict):
            qty      = int(data.get("qty", 0))
            entry_px = float(data.get("entry_price", 0))
        else:
            qty      = int(data or 0)
            entry_px = 0.0

        if qty <= 0:
            continue

        current_px          = float(prices.get(ticker, entry_px))
        total_market_value += qty * current_px

    net_worth        = cash + total_market_value
    original_capital = float(cfg.get("original_capital", 100000.0))

    if original_capital <= 0:
        return 0.0

    return (net_worth - original_capital) / original_capital
    # FIX 1: Dead code that was here previously (old entry-price formula) is removed.


def score_positions(portfolio: dict, prices: pd.Series) -> pd.DataFrame:
    """
    DataFrame of open positions ranked worst->best by per-stock return vs
    their individual entry prices (not overall portfolio return).
    Columns: ticker, return, qty, entry_price, current_price, market_value
    """
    holdings = portfolio.get("holdings", {})
    rows = []
    for ticker, data in holdings.items():
        if isinstance(data, dict):
            qty      = int(data.get("qty", 0))
            entry_px = float(data.get("entry_price", 0))
        else:
            qty      = int(data or 0)
            entry_px = 0.0

        if qty <= 0:
            continue

        current_px   = float(prices.get(ticker, entry_px))
        ret          = (current_px - entry_px) / entry_px if entry_px > 0 else 0.0
        market_value = qty * current_px

        rows.append({
            "ticker":        ticker,
            "return":        ret,
            "qty":           qty,
            "entry_price":   entry_px,
            "current_price": current_px,
            "market_value":  market_value,
        })

    return pd.DataFrame(rows).sort_values("return").reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# 4. Candidate selection
# ════════════════════════════════════════════════════════════════════════════

def get_strong_candidates(
    params: dict,
    exclude_tickers: list[str],
    min_score: float,
) -> list[dict]:
    """
    Pull tickers from optimal_params.json with stability_score >= min_score
    that are not already held. Sorted best->worst by stability_score.
    """
    candidates = []
    for ticker, plan in params.items():
        if ticker in exclude_tickers:
            continue
        if plan.get("strategy") == "NONE":
            continue
        score = float(plan.get("stability_score", 0) or 0)
        if score >= min_score:
            candidates.append({
                "ticker":          ticker,
                "stability_score": score,
                "strategy":        plan.get("strategy", ""),
                "expected_return": plan.get("expected_return", 0),
            })
    return sorted(candidates, key=lambda x: -x["stability_score"])


# ════════════════════════════════════════════════════════════════════════════
# 5. Regime & macro gates  (mirrors daily_screener.py exactly)
# ════════════════════════════════════════════════════════════════════════════

def check_regime_and_macro(mode: str, market: str, mo: dict):
    """
    Returns: is_uptrend, allow_buys, nifty_drop, macro, macro_context, regime_str
    allow_buys=False means sell weak but hold freed capital as cash.
    """
    is_uptrend, nifty_close, nifty_ema = get_market_regime()
    nifty_drop = 0.0

    if not is_uptrend:
        nifty_drop = get_nifty_drop_from_50d_high()

    buy_in_downtrend = mo.get("buy_in_downtrend", False)
    nifty_threshold  = mo.get("nifty_dip_pct_required", 0.05)

    if is_uptrend:
        allow_buys = True
    elif buy_in_downtrend and nifty_drop >= nifty_threshold:
        allow_buys = True
    else:
        allow_buys = False

    trend_str  = "UPTREND ✅" if is_uptrend else f"DOWNTREND ❌ (drop {nifty_drop:.1%})"
    regime_str = f"Nifty {nifty_close:.0f} vs EMA50 {nifty_ema:.0f} -> {trend_str}"

    macro = macro_context = None
    if market in MARKET_CONFIGS:
        macro         = MacroFilter(MARKET_CONFIGS[market])
        macro_context = macro.run()
        macro.print_summary(macro_context)

        if macro_context.is_high_risk_day:
            allow_buys = False
            print(f"  ⚠️  HIGH-RISK DAY [{macro_context.high_risk_reason}] — buys suppressed.")
    else:
        print(f"  ⚠️  Unknown market '{market}' — macro filter skipped.")

    return is_uptrend, allow_buys, nifty_drop, macro, macro_context, regime_str


# ════════════════════════════════════════════════════════════════════════════
# 6. Rotation plan builder
# ════════════════════════════════════════════════════════════════════════════

def build_rotation_plan(
    scored_positions: pd.DataFrame,
    candidates: list[dict],
    capital_freed: float,
    n_slots: int,
    prices: pd.Series,
    params: dict,
    macro,
    macro_context,
    allow_buys: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (sells, buys). Buys sized by ATR — same formula as daily_screener.
    """
    sells = []
    for _, row in scored_positions.iterrows():
        sells.append({
            "ticker":        row["ticker"],
            "qty":           int(row["qty"]),
            "current_price": round(row["current_price"], 2),
            "return_pct":    round(row["return"] * 100, 2),
            "market_value":  round(row["market_value"], 2),
            "reason":        f"ProfitChaser:Weak ({row['return']*100:.1f}%)",
        })

    buys = []
    if not allow_buys or not candidates or capital_freed <= 0:
        return sells, buys

    alloc_per = capital_freed / max(n_slots, 1)
    added     = 0

    for c in candidates:
        if added >= n_slots:
            break

        ticker = c["ticker"]

        # Per-ticker macro gate — same as daily_screener Gate 1b
        if macro is not None and macro_context is not None:
            eval_result = macro.evaluate_buy(ticker, macro_context)
            if eval_result.action == FilterAction.SKIP:
                print(f"  🌍 Macro BLOCKED {ticker}: {eval_result.reason}")
                continue

        px = float(prices.get(ticker, 0))
        if px <= 0:
            continue

        # ATR-based qty sizing
        try:
            df_hist = yf.download(ticker, period="60d", progress=False, auto_adjust=True)
            if df_hist.empty or len(df_hist) < 15:
                qty = max(int(alloc_per / px), 1)
            else:
                atr = calculate_atr(df_hist)
                if atr > 0:
                    atr_qty    = int((alloc_per * 0.01) / (atr * 2))
                    budget_qty = int(alloc_per / px)
                    qty        = max(min(atr_qty, budget_qty), 1)
                else:
                    qty = max(int(alloc_per / px), 1)
        except Exception:
            qty = max(int(alloc_per / px), 1)

        cost = qty * px
        if cost > alloc_per * 1.05:
            qty  = max(int(alloc_per / px), 1)
            cost = qty * px

        buys.append({
            "ticker":          ticker,
            "qty":             qty,
            "current_price":   round(px, 2),
            "estimated_cost":  round(cost, 2),
            "stability_score": round(c["stability_score"], 1),
            "strategy":        c["strategy"],
            "reason":          f"ProfitChaser:Strong (score {c['stability_score']:.1f}%)",
        })
        added += 1

    return sells, buys


# ════════════════════════════════════════════════════════════════════════════
# 7. Reporting
# ════════════════════════════════════════════════════════════════════════════

def print_report(
    current_return: float,
    target_pct: float,
    gap: float,
    urgency: float,
    days_remaining: int,
    mode: str,
    regime_str: str,
    allow_buys: bool,
    sells: list[dict],
    buys: list[dict],
    dry_run: bool,
):
    sep = "=" * 70
    print(f"\n{sep}")
    print("🎯  T_RAIDER PROFIT CHASER")
    print(sep)
    print(f"  Mode          : {mode}")
    print(f"  Regime        : {regime_str}")
    print(f"  Quarter target: {target_pct:.1%}  |  Current return : {current_return:.2%}")
    print(f"  Gap to target : {gap:.2%}          |  Days left      : {days_remaining}")
    print(f"  Urgency mult  : {urgency:.1f}x")
    print(f"  Buys allowed  : {'YES' if allow_buys else 'NO (downtrend / high-risk day)'}")
    print(f"  Run type      : {'🔍 DRY RUN — no orders placed' if dry_run else '🚀 LIVE EXECUTION'}")
    print(sep)

    if not sells:
        print("\n  ✅ No weak positions to rotate out.")
    else:
        print(f"\n  📉 SELL ({len(sells)} position{'s' if len(sells)>1 else ''}):")
        print(f"  {'Ticker':12} {'Qty':>6} {'Price':>10} {'Return':>8} {'Value':>12}")
        print("  " + "-" * 54)
        for s in sells:
            print(
                f"  {s['ticker']:12} {s['qty']:>6} "
                f"₹{s['current_price']:>8.2f} {s['return_pct']:>+7.1f}% "
                f"₹{s['market_value']:>10,.0f}"
            )

    if not buys:
        if allow_buys:
            print("\n  ℹ️  No qualified replacement candidates found.")
        else:
            print("\n  💵  Capital freed — held as cash (downtrend / high-risk mode).")
    else:
        print(f"\n  📈 BUY ({len(buys)} position{'s' if len(buys)>1 else ''}):")
        print(f"  {'Ticker':12} {'Qty':>6} {'Price':>10} {'Cost':>12} {'Score':>7}")
        print("  " + "-" * 54)
        for b in buys:
            print(
                f"  {b['ticker']:12} {b['qty']:>6} "
                f"₹{b['current_price']:>8.2f} "
                f"₹{b['estimated_cost']:>10,.0f} "
                f"{b['stability_score']:>6.1f}%"
            )

    total_sell_val = sum(s["market_value"] for s in sells)
    total_buy_cost = sum(b["estimated_cost"] for b in buys)
    print(f"\n  Capital freed : ₹{total_sell_val:>12,.0f}")
    print(f"  Capital used  : ₹{total_buy_cost:>12,.0f}")
    print(f"  Residual cash : ₹{total_sell_val - total_buy_cost:>12,.0f}")
    print(sep)


def log_chaser_run(
    mode: str,
    current_return: float,
    target_pct: float,
    gap: float,
    sells: list[dict],
    buys: list[dict],
    dry_run: bool,
):
    CHASER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(CHASER_LOG_PATH.read_text()) if CHASER_LOG_PATH.exists() else []
    except Exception:
        existing = []

    existing.append({
        "timestamp":      datetime.now().isoformat(),
        "mode":           mode,
        "dry_run":        dry_run,
        "current_return": round(current_return * 100, 3),
        "target_pct":     round(target_pct * 100, 3),
        "gap_pct":        round(gap * 100, 3),
        "sells":          sells,
        "buys":           buys,
    })

    CHASER_LOG_PATH.write_text(json.dumps(existing, indent=2))
    print(f"\n  📁 Chaser log updated -> {CHASER_LOG_PATH}")


# ════════════════════════════════════════════════════════════════════════════
# 8. Paper-trade execution
# ════════════════════════════════════════════════════════════════════════════

def execute_paper(portfolio: dict, sells: list[dict], buys: list[dict]) -> dict:
    """Simulate execution in paper mode — no Upstox call needed."""
    holdings = portfolio.setdefault("holdings", {})
    cash     = float(portfolio.get("cash", 0))

    for s in sells:
        ticker = s["ticker"]
        if ticker in holdings:
            cash += s["market_value"]
            del holdings[ticker]
            print(f"  [PAPER SELL] {ticker} x{s['qty']} @ ₹{s['current_price']} -> freed ₹{s['market_value']:,.0f}")

    for b in buys:
        ticker = b["ticker"]
        cost   = b["estimated_cost"]
        if cash >= cost:
            holdings[ticker] = {
                "qty":         b["qty"],
                "entry_price": b["current_price"],
            }
            cash -= cost
            print(f"  [PAPER BUY]  {ticker} x{b['qty']} @ ₹{b['current_price']} -> cost ₹{cost:,.0f}")
        else:
            print(
                f"  [PAPER BUY]  {ticker} SKIPPED — insufficient cash "
                f"(need ₹{cost:,.0f}, have ₹{cash:,.0f})"
            )

    portfolio["cash"] = round(cash, 2)
    return portfolio


# ════════════════════════════════════════════════════════════════════════════
# 9. Main entry point
# ════════════════════════════════════════════════════════════════════════════

def _stamp_and_save(cfg: dict) -> None:
    """FIX 5: Stamp last_chaser_run and save on ALL exit paths."""
    cfg["chaser"]["last_chaser_run"] = datetime.now().isoformat()
    save_quarterly_config(cfg)


def run_chaser(mode: str = "CONSERVATIVE", market: str = "INDIA", dry_run: bool = True):
    print(f"\n{'─'*70}")
    print(f"  T_Raider Profit Chaser  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*70}")

    # ── Load configs ────────────────────────────────────────────────────────
    cfg    = load_quarterly_config()
    chaser = _chaser_cfg(cfg)          # live reference into cfg["chaser"]
    params = load_optimal_params()

    if not chaser.get("enabled", True):
        print("  ℹ️  Profit chaser is disabled in quarterly_config.json -> chaser.enabled")
        return

    # Two locks: CLI --execute flag AND config chaser.dry_run must both allow
    effective_dry_run = dry_run or chaser.get("dry_run", True)

    target_pct       = float(cfg.get("profit_target_pct", 0.05))
    max_rotation_pct = float(chaser.get("max_rotation_pct", 0.30))
    gap_trigger_pct  = float(chaser.get("gap_trigger_pct", 0.40))

    mo      = _mode_overrides(chaser, mode)
    urgency = urgency_multiplier(cfg, chaser)

    # Higher urgency -> less negative threshold -> more stocks qualify as weak
    effective_weak_threshold = mo["weak_threshold"] / urgency

    print(f"  Target         : {target_pct:.1%}")
    print(f"  Urgency        : {urgency:.1f}x  (effective weak threshold: {effective_weak_threshold:.2%})")
    print(f"  Days left      : {days_left(cfg)}")
    print(f"  Effective mode : {mode}")
    print(f"  Dry run        : {effective_dry_run}")

    # ── Load portfolio & prices ─────────────────────────────────────────────
    portfolio    = load_portfolio()
    held_tickers = list(portfolio.get("holdings", {}).keys())

    if not held_tickers:
        print("\n  ℹ️  Portfolio is empty — nothing to rotate.")
        _stamp_and_save(cfg)
        return

    print(f"\n  Fetching prices for {len(held_tickers)} holdings…")
    prices = fetch_latest_prices(held_tickers)

    # ── Current return vs target ─────────────────────────────────────────────
    current_return = portfolio_current_return(portfolio, prices, cfg)
    gap            = target_pct - current_return

    print(f"\n  Portfolio return : {current_return:.2%}")
    print(f"  Target           : {target_pct:.1%}")
    print(f"  Gap              : {gap:.2%}")

    if gap <= 0:
        print("\n  🎉 Target already met — no rotation needed.")
        _stamp_and_save(cfg)
        return

    if (gap / target_pct) < gap_trigger_pct:
        print(
            f"\n  ✅ Gap ({gap:.2%}) is less than {gap_trigger_pct:.0%} of target "
            f"({target_pct * gap_trigger_pct:.2%}) — close enough, no rotation."
        )
        _stamp_and_save(cfg)
        return

    # ── Regime + macro gates ─────────────────────────────────────────────────
    print("\n  Checking regime & macro gates…")
    is_uptrend, allow_buys, nifty_drop, macro, macro_context, regime_str = \
        check_regime_and_macro(mode, market, mo)

    # ── Score existing positions ─────────────────────────────────────────────
    scored = score_positions(portfolio, prices)
    weak   = scored[scored["return"] < effective_weak_threshold].copy()

    if weak.empty:
        print(
            f"\n  ✅ No positions below weak threshold ({effective_weak_threshold:.2%}). "
            f"Nothing to sell."
        )
        _stamp_and_save(cfg)
        return

    # ── Cap rotation at max_rotation_pct of total portfolio value ────────────
    total_mkt_val    = scored["market_value"].sum() + float(portfolio.get("cash", 0))
    max_rotate_value = total_mkt_val * max_rotation_pct

    weak["cumulative_value"] = weak["market_value"].cumsum()
    weak_trimmed = weak[weak["cumulative_value"] <= max_rotate_value]
    if weak_trimmed.empty:
        weak_trimmed = weak.head(1)   # always sell at least the single weakest

    capital_freed = float(weak_trimmed["market_value"].sum())
    n_slots       = len(weak_trimmed)

    # ── Strong candidates from optimal_params ────────────────────────────────
    min_score  = mo["strong_min_score"]
    candidates = get_strong_candidates(params, held_tickers, min_score)

    if not candidates and allow_buys:
        print(f"\n  ⚠️  No candidates with stability >= {min_score:.0%} found outside portfolio.")

    candidate_tickers = [c["ticker"] for c in candidates[: n_slots * 2]]
    if candidate_tickers:
        print(f"\n  Fetching prices for {len(candidate_tickers)} candidate stocks…")
        cand_prices = fetch_latest_prices(candidate_tickers)
        prices = pd.concat([prices, cand_prices[~cand_prices.index.isin(prices.index)]])

    # ── Build rotation plan ──────────────────────────────────────────────────
    sells, buys = build_rotation_plan(
        scored_positions=weak_trimmed,
        candidates=candidates,
        capital_freed=capital_freed,
        n_slots=n_slots,
        prices=prices,
        params=params,
        macro=macro,
        macro_context=macro_context,
        allow_buys=allow_buys,
    )

    # ── Print report ─────────────────────────────────────────────────────────
    print_report(
        current_return=current_return,
        target_pct=target_pct,
        gap=gap,
        urgency=urgency,
        days_remaining=days_left(cfg),
        mode=mode,
        regime_str=regime_str,
        allow_buys=allow_buys,
        sells=sells,
        buys=buys,
        dry_run=effective_dry_run,
    )

    # ── Execute ──────────────────────────────────────────────────────────────
    if not effective_dry_run and (sells or buys):
        paper_trading = cfg.get("paper_trading", True)
        broker        = cfg.get("broker", "paper")

        if paper_trading or broker == "paper":
            print("\n  📝 Paper trading mode — simulating execution…")
            updated_portfolio = execute_paper(portfolio, sells, buys)
            portfolio_path = Path("config/portfolio.json")
            if portfolio_path.exists():
                portfolio_path.write_text(json.dumps(updated_portfolio, indent=2))
                print("  ✅ portfolio.json updated.")
        else:
            print(f"\n  🔴 LIVE broker '{broker}' — integrate with your execution adapter.")
            print("  Sells:", [s["ticker"] for s in sells])
            print("  Buys: ", [b["ticker"] for b in buys])

    # ── FIX 2: Use live cfg["chaser"] reference — NOT setdefault ─────────────
    _stamp_and_save(cfg)

    # ── Persist run log ───────────────────────────────────────────────────────
    log_chaser_run(
        mode=mode,
        current_return=current_return,
        target_pct=target_pct,
        gap=gap,
        sells=sells,
        buys=buys,
        dry_run=effective_dry_run,
    )


# ════════════════════════════════════════════════════════════════════════════
# 10. CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="T_Raider Profit Chaser v1.1 — actively rotate portfolio toward quarterly target"
    )
    parser.add_argument(
        "--mode",
        choices=["CONSERVATIVE", "BALANCED", "AGGRESSIVE"],
        default="CONSERVATIVE",
        help="Risk mode (default: CONSERVATIVE)",
    )
    parser.add_argument(
        "--market",
        choices=list(MARKET_CONFIGS.keys()),
        default="INDIA",
        help="Target market for macro/VIX checks (default: INDIA)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute trades. Without this flag the run is always a dry run.",
    )
    args = parser.parse_args()

    run_chaser(
        mode=args.mode,
        market=args.market,
        dry_run=not args.execute,
    )