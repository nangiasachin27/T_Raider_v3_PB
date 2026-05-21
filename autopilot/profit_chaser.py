"""
autopilot/profit_chaser.py  (v1.0)
────────────────────────────────────────────────────────────────────────────────
Active Profit Chaser — bridges the gap between current portfolio return and the
quarterly_config.json target by rotating weak→strong holdings.

Unlike the passive profit-booking system (which waits for target% to be hit or
time period to expire), the chaser ACTS:
  • Identifies lagging positions (below weak_threshold return)
  • Sells them to free capital
  • Redeploys that capital into stronger screener candidates
  • Respects the CONSERVATIVE / BALANCED / AGGRESSIVE mode gate,
    including Nifty regime and macro checks already used by daily_screener.py

Usage (manual test):
    # Dry run — safe, prints plan only, touches nothing
    python -m autopilot.profit_chaser --mode CONSERVATIVE

    # See what BALANCED mode would do
    python -m autopilot.profit_chaser --mode BALANCED

    # Execute (paper or live, controlled by quarterly_config.json → broker)
    python -m autopilot.profit_chaser --mode CONSERVATIVE --execute

Architecture:
    quarterly_config.json          ← target_pct, chaser block, broker flag
    config/optimal_params.json     ← strong candidate universe (Monte Carlo)
    autopilot/logger.py            ← load_portfolio() — same as daily_screener
    ingestion/nse_constituents.py  ← get_market_regime() — same Gate 1
    macro_filter.py                ← MacroFilter — same Gate 1b
    daily_screener.py              ← calculate_atr(), get_nifty_drop_from_50d_high()
"""

from __future__ import annotations

import json
import warnings
import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# ── Internal imports — same modules daily_screener uses ──────────────────────
from autopilot.logger import load_portfolio
from ingestion.nse_constituents import get_market_regime
from daily_screener import calculate_atr, get_nifty_drop_from_50d_high
from macro_filter import MacroFilter, MARKET_CONFIGS, FilterAction

# ── Config paths ─────────────────────────────────────────────────────────────
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
    with open(QUARTERLY_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def load_optimal_params() -> dict:
    if not OPTIMAL_PARAMS_PATH.exists():
        raise FileNotFoundError(
            "config/optimal_params.json not found. Run auto_optimizer.py first."
        )
    with open(OPTIMAL_PARAMS_PATH) as f:
        return json.load(f)


def _chaser_cfg(cfg: dict) -> dict:
    """Return the chaser sub-block with safe defaults."""
    return cfg.get("chaser", {})


def _mode_overrides(chaser: dict, mode: str) -> dict:
    """Merge base chaser defaults with the selected mode's overrides."""
    base = {
        "weak_threshold":   chaser.get("weak_threshold",   -0.05),
        "strong_min_score": chaser.get("strong_min_score",  0.70),
        "buy_in_downtrend": False,
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
    Returns 1.0 (normal) → 1.5 (moderate) → 2.0 (last stretch).
    Only active when chaser.urgency_scaling.enabled is true.
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
    """Download latest close prices for all tickers in one batch call."""
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
    Mirrors tracker.py exactly:
      gross_pl    = (cash + market_value_of_holdings) - original_capital
      return_pct  = gross_pl / original_capital
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
    """
    Weighted portfolio return vs entry prices.
    Uses the same holdings structure that load_portfolio() returns.
    """
    holdings = portfolio.get("holdings", {})
    total_cost = total_value = 0.0

    for ticker, data in holdings.items():
        if isinstance(data, dict):
            qty        = int(data.get("qty", 0))
            entry_px   = float(data.get("entry_price", 0))
        else:
            qty        = int(data or 0)
            entry_px   = 0.0

        if qty <= 0 or entry_px <= 0:
            continue

        current_px = float(prices.get(ticker, entry_px))
        total_cost  += qty * entry_px
        total_value += qty * current_px

    if total_cost <= 0:
        return 0.0
    return (total_value - total_cost) / total_cost


def score_positions(portfolio: dict, prices: pd.Series) -> pd.DataFrame:
    """
    Returns a DataFrame of open positions ranked worst→best by current return.
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
    Pull tickers from optimal_params.json that:
      • have stability_score >= min_score
      • are NOT already held in the portfolio
    Returns sorted best→worst by stability_score.
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
                "ticker":         ticker,
                "stability_score": score,
                "strategy":       plan.get("strategy", ""),
                "expected_return": plan.get("expected_return", 0),
            })
    return sorted(candidates, key=lambda x: -x["stability_score"])


# ════════════════════════════════════════════════════════════════════════════
# 5. Regime & macro gates  (mirrors daily_screener.py exactly)
# ════════════════════════════════════════════════════════════════════════════

def check_regime_and_macro(mode: str, market: str, mo: dict):
    """
    Returns:
        is_uptrend         bool
        allow_buys         bool  — False means sell weak but hold cash
        nifty_drop         float
        macro              MacroFilter | None
        macro_context      MacroContext | None
        regime_summary_str str
    """
    # Gate 1 — Market Regime (same call as daily_screener)
    is_uptrend, nifty_close, nifty_ema = get_market_regime()
    nifty_drop = 0.0

    if not is_uptrend:
        nifty_drop = get_nifty_drop_from_50d_high()

    # Determine whether buys are allowed given mode + regime
    buy_in_downtrend = mo.get("buy_in_downtrend", False)
    nifty_threshold  = mo.get("nifty_dip_pct_required", 0.05)

    if is_uptrend:
        allow_buys = True
    elif buy_in_downtrend and nifty_drop >= nifty_threshold:
        allow_buys = True   # BALANCED/AGGRESSIVE with sufficient dip
    else:
        allow_buys = False  # CONSERVATIVE or dip not big enough

    trend_str = "UPTREND ✅" if is_uptrend else f"DOWNTREND ❌ (drop {nifty_drop:.1%})"
    regime_str = f"Nifty {nifty_close:.0f} vs EMA50 {nifty_ema:.0f} → {trend_str}"

    # Gate 1b — Macro Environment (same MacroFilter as daily_screener)
    macro = macro_context = None
    if market in MARKET_CONFIGS:
        macro         = MacroFilter(MARKET_CONFIGS[market])
        macro_context = macro.run()
        macro.print_summary(macro_context)

        # If macro says SKIP for any global reason (extreme VIX, overnight crash,
        # earnings blackout on the portfolio level), suppress buys entirely.
        # Note: evaluate_buy() is per-ticker in the buy loop below, not here.
        # Here we only check for a portfolio-wide block.
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
    Returns (sells, buys) as lists of dicts describing the rotation.
    Buys are sized by ATR using the same logic as daily_screener.
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
    added = 0

    for c in candidates:
        if added >= n_slots:
            break

        ticker = c["ticker"]

        # Per-ticker macro gate — same as daily_screener BUY pipeline Gate 1b
        if macro is not None and macro_context is not None:
            eval_result = macro.evaluate_buy(ticker, macro_context)
            if eval_result.action == FilterAction.SKIP:
                print(f"  🌍 Macro BLOCKED {ticker}: {eval_result.reason}")
                continue

        # ATR-based qty (same formula as daily_screener)
        px = float(prices.get(ticker, 0))
        if px <= 0:
            continue

        # Fetch recent data for ATR calculation
        try:
            df_hist = yf.download(ticker, period="60d", progress=False, auto_adjust=True)
            if df_hist.empty or len(df_hist) < 15:
                qty = max(int(alloc_per / px), 1)
            else:
                atr = calculate_atr(df_hist)
                # ATR sizing: 1% of alloc / (ATR × 2), but also capped by alloc_per
                if atr > 0:
                    atr_qty = int((alloc_per * 0.01) / (atr * 2))
                    budget_qty = int(alloc_per / px)
                    qty = max(min(atr_qty, budget_qty), 1)
                else:
                    qty = max(int(alloc_per / px), 1)
        except Exception:
            qty = max(int(alloc_per / px), 1)

        cost = qty * px
        if cost > alloc_per * 1.05:      # allow 5% overshoot
            qty = max(int(alloc_per / px), 1)
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
            print("\n  💵  Capital freed — held as cash (downtrend mode).")
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
    print(f"\n  📁 Chaser log updated → {CHASER_LOG_PATH}")


# ════════════════════════════════════════════════════════════════════════════
# 8. Paper-trade execution (no real broker call needed in paper mode)
# ════════════════════════════════════════════════════════════════════════════

def execute_paper(portfolio: dict, sells: list[dict], buys: list[dict]) -> dict:
    """
    Simulates execution against the in-memory portfolio dict.
    In paper mode this is all we need — no Upstox call.
    Returns updated portfolio dict.
    """
    holdings = portfolio.setdefault("holdings", {})
    cash     = float(portfolio.get("cash", 0))

    for s in sells:
        ticker = s["ticker"]
        if ticker in holdings:
            cash += s["market_value"]
            del holdings[ticker]
            print(f"  [PAPER SELL] {ticker} ×{s['qty']} @ ₹{s['current_price']} → freed ₹{s['market_value']:,.0f}")

    for b in buys:
        ticker = b["ticker"]
        cost   = b["estimated_cost"]
        if cash >= cost:
            holdings[ticker] = {
                "qty":         b["qty"],
                "entry_price": b["current_price"],
            }
            cash -= cost
            print(f"  [PAPER BUY]  {ticker} ×{b['qty']} @ ₹{b['current_price']} → cost ₹{cost:,.0f}")
        else:
            print(f"  [PAPER BUY]  {ticker} SKIPPED — insufficient cash (need ₹{cost:,.0f}, have ₹{cash:,.0f})")

    portfolio["cash"] = round(cash, 2)
    return portfolio


# ════════════════════════════════════════════════════════════════════════════
# 9. Main entry point
# ════════════════════════════════════════════════════════════════════════════

def run_chaser(mode: str = "CONSERVATIVE", market: str = "INDIA", dry_run: bool = True):
    print(f"\n{'─'*70}")
    print(f"  T_Raider Profit Chaser  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*70}")

    # ── Load configs ────────────────────────────────────────────────────────
    cfg     = load_quarterly_config()
    chaser  = _chaser_cfg(cfg)
    params  = load_optimal_params()

    if not chaser.get("enabled", True):
        print("  ℹ️  Profit chaser is disabled in quarterly_config.json → chaser.enabled")
        return

    # Honour the config-level dry_run as a hard lock (two locks policy)
    effective_dry_run = dry_run or chaser.get("dry_run", True)

    target_pct       = float(cfg.get("profit_target_pct", 0.05))
    max_rotation_pct = float(chaser.get("max_rotation_pct", 0.30))
    gap_trigger_pct  = float(chaser.get("gap_trigger_pct", 0.40))

    # Mode overrides (thresholds from config, not hardcoded)
    mo = _mode_overrides(chaser, mode)

    # Apply urgency multiplier to weak_threshold
    # Higher urgency → less negative threshold → more stocks qualify as weak
    urgency = urgency_multiplier(cfg, chaser)
    effective_weak_threshold = mo["weak_threshold"] / urgency   # e.g. -0.05 / 1.5 = -0.033

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
        return

    print(f"\n  Fetching prices for {len(held_tickers)} holdings…")
    prices = fetch_latest_prices(held_tickers)

    # ── Current return vs target ─────────────────────────────────────────────
    current_return = portfolio_current_return(portfolio, prices, cfg)
    gap            = target_pct - current_return

    print(f"\n  Portfolio return : {current_return:.2%}")
    print(f"  Target           : {target_pct:.1%}")
    print(f"  Gap              : {gap:.2%}")

    # Only act if the remaining gap is meaningful
    if gap <= 0:
        print("\n  🎉 Target already met — no rotation needed.")
        return

    if (gap / target_pct) < gap_trigger_pct:
        print(
            f"\n  ✅ Gap ({gap:.2%}) is less than {gap_trigger_pct:.0%} of target "
            f"({target_pct * gap_trigger_pct:.2%}) — close enough, no rotation."
        )
        return

    # ── Regime + macro gates ─────────────────────────────────────────────────
    print("\n  Checking regime & macro gates…")
    is_uptrend, allow_buys, nifty_drop, macro, macro_context, regime_str = \
        check_regime_and_macro(mode, market, mo)

    # ── Score existing positions ─────────────────────────────────────────────
    scored = score_positions(portfolio, prices)

    # Identify weak stocks (below effective threshold)
    weak = scored[scored["return"] < effective_weak_threshold].copy()

    if weak.empty:
        print(
            f"\n  ✅ No positions below weak threshold ({effective_weak_threshold:.2%}). "
            f"Nothing to sell."
        )
        return

    # Cap rotation at MAX_ROTATION_PCT of total portfolio value
    portfolio_value  = float(prices.apply(lambda px: 0).sum())  # recalculate properly
    total_mkt_val    = scored["market_value"].sum() + float(portfolio.get("cash", 0))
    max_rotate_value = total_mkt_val * max_rotation_pct

    # Trim weak list to respect rotation cap
    weak = weak.copy()
    weak["cumulative_value"] = weak["market_value"].cumsum()
    weak_trimmed = weak[weak["cumulative_value"] <= max_rotate_value]
    if weak_trimmed.empty:
        # At least sell the single weakest position
        weak_trimmed = weak.head(1)

    capital_freed = float(weak_trimmed["market_value"].sum())
    n_slots       = len(weak_trimmed)

    # ── Strong candidates from optimal_params ────────────────────────────────
    min_score   = mo["strong_min_score"]
    candidates  = get_strong_candidates(params, held_tickers, min_score)

    if not candidates and allow_buys:
        print(f"\n  ⚠️  No candidates with stability ≥ {min_score:.0%} found outside portfolio.")

    # Fetch prices for candidates (top 2× slots to have fallbacks)
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

    # ── Execute or log ───────────────────────────────────────────────────────
    if not effective_dry_run and (sells or buys):
        paper_trading = cfg.get("paper_trading", True)
        broker        = cfg.get("broker", "paper")

        if paper_trading or broker == "paper":
            print("\n  📝 Paper trading mode — simulating execution…")
            updated_portfolio = execute_paper(portfolio, sells, buys)
            # Persist changes back through the same file load_portfolio uses
            portfolio_path = Path("config/portfolio.json")
            if portfolio_path.exists():
                portfolio_path.write_text(json.dumps(updated_portfolio, indent=2))
                print(f"  ✅ portfolio.json updated.")
        else:
            # Live broker path — wire to your Upstox adapter here
            print(f"\n  🔴 LIVE broker '{broker}' — integrate with your execution adapter.")
            print("  Sells:", [s["ticker"] for s in sells])
            print("  Buys: ", [b["ticker"] for b in buys])

    # ── Update last_chaser_run in config ─────────────────────────────────────
    cfg.setdefault("chaser", {})["last_chaser_run"] = datetime.now().isoformat()
    save_quarterly_config(cfg)

    # ── Persist log ──────────────────────────────────────────────────────────
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
        description="T_Raider Profit Chaser v1.0 — actively rotate portfolio toward quarterly target"
    )
    parser.add_argument(
        "--mode",
        choices=["CONSERVATIVE", "BALANCED", "AGGRESSIVE"],
        default="CONSERVATIVE",
        help="Risk mode. Controls which regime/macro gates apply and how aggressively weak stocks are sold. (default: CONSERVATIVE)",
    )
    parser.add_argument(
        "--market",
        choices=list(MARKET_CONFIGS.keys()),
        default="INDIA",
        help="Target market — controls VIX source and macro calendar. (default: INDIA)",
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