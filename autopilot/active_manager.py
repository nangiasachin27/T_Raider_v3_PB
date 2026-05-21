"""
autopilot/active_manager.py
────────────────────────────
Active Profit Engineering v2 — Optimized for aggressive target pursuit.

Reads quarterly_config.json (profit_target_pct, base capital, quarter dates).
Reads optimal_params.json (per-stock expected_return, stability, sharpe).
Reads stocks.json (full universe for proactive opportunity scanning).
Reads portfolio.json (holdings, entry prices, dates).

NO new config files created.

Three mechanisms (enhanced):
  1. Portfolio-Level Milestone Check — compares actual vs expected trajectory
  2. Proactive Rebalancing — scans FULL universe, ranks all opportunities
  3. Dynamic Risk Scaling — adjusts position sizes based on distance to target
"""

import json
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autopilot.logger import load_portfolio, record_transaction, _normalise_holding


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — EDIT HERE ONLY. All parameters in one place.
# ═══════════════════════════════════════════════════════════════════════════════

class ActiveConfig:
    """All active-engine tunable parameters. No external files needed."""

    # --- Milestone thresholds ---
    AHEAD_MULTIPLIER = 1.5      # actual >= expected * 1.5 → LOCK_IN (tighten)
    BEHIND_MULTIPLIER = 0.5     # actual < expected * 0.5 → PUSH (aggressive)

    # --- Trim winners ---
    TRIM_GAIN_THRESHOLD = 0.10   # trim if unrealized gain >= +15%
    TRIM_FRACTION = 0.50         # sell 30% of position
    TRIM_MAX_PER_RUN = 2         # trim max 2 winners per run

    # --- Cut losers ---
    CUT_LOSS_THRESHOLD = -0.03   # consider cutting if unrealized loss <= -5%
    CUT_AGGRESSIVE_THRESHOLD = -0.03  # cut deeper when behind target
    CUT_MAX_PER_RUN = 2          # max losers to cut per run
    CUT_MAX_AGGRESSIVE = 3       # max losers when behind target

    # --- Dead money ---
    DEAD_MONEY_DAYS = 45         # max days without meaningful progress
    DEAD_MONEY_MIN_GAIN = 0.02   # must be +2% after DEAD_MONEY_DAYS

    # --- Risk scaling ---
    RISK_AGGRESSIVE = 1.5        # behind target → 1.5x position sizing
    RISK_DEFENSIVE = 0.8         # near target → 0.8x position sizing
    RISK_NORMAL = 1.0

    # --- Proactive rebalancing ---
    REBALANCE_MIN_GAP = 20       # new opportunity must score >= holding_score + 20
    REBALANCE_MAX_POSITIONS = 8  # target max open positions
    REBALANCE_CASH_BUFFER = 5000   # always keep ₹5,000 cash

    # --- Target derivation ---
    TARGET_FROM_EXPECTED_RETURN = True  # use optimal_params expected_return * 0.5 as per-stock target
    TARGET_CAP_MIN = 0.15        # minimum per-stock target +10%
    TARGET_CAP_MAX = 0.50        # maximum per-stock target +50%


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS (read existing files only)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_quarterly_config() -> Dict:
    """Read config/quarterly_config.json — single source of truth."""
    path = Path("config/quarterly_config.json")
    if not path.exists():
        raise FileNotFoundError("config/quarterly_config.json not found")
    with open(path) as f:
        return json.load(f)


def _load_optimal_params() -> Dict:
    """Read config/optimal_params.json — per-stock expected returns."""
    path = Path("config/optimal_params.json")
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_stocks_universe() -> List[str]:
    """Read config/stocks.json — full universe for proactive scanning."""
    path = Path("config/stocks.json")
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("nifty_50", [])


def _get_quarter_start(cfg: Dict) -> date:
    if "current_base_capital_date" in cfg:
        return datetime.strptime(cfg["current_base_capital_date"], "%Y-%m-%d").date()
    if "quarter_start" in cfg:
        return datetime.strptime(cfg["quarter_start"], "%Y-%m-%d").date()
    portfolio = load_portfolio()
    history = portfolio.get("history", [])
    if history:
        first_trade = datetime.strptime(history[0]["timestamp"].split()[0], "%Y-%m-%d")
        return first_trade.date()
    return date.today()


def _get_base_capital(cfg: Dict) -> float:
    if "current_base_capital" in cfg:
        return float(cfg["current_base_capital"])
    if "total_baseline_wealth" in cfg:
        return float(cfg["total_baseline_wealth"])
    portfolio = load_portfolio()
    return portfolio.get("cash", 100000.0)


def _get_target_pct(cfg: Dict) -> float:
    return float(cfg.get("profit_target_pct", 0.05))


def _get_quarter_days(cfg: Dict) -> int:
    return int(cfg.get("quarter_days", 90))


# ═══════════════════════════════════════════════════════════════════════════════
# PER-STOCK TARGET (from optimal_params)
# ═══════════════════════════════════════════════════════════════════════════════

def get_stock_target_pct(ticker: str, optimal_params: Dict) -> float:
    """
    Derive per-stock profit target from optimal_params expected_return.
    If expected_return is 0 or missing, falls back to quarterly target.
    """
    cfg = ActiveConfig
    plan = optimal_params.get(ticker, {})
    expected = plan.get("expected_return", 0)

    if expected > 0 and cfg.TARGET_FROM_EXPECTED_RETURN:
        # Use half of expected return as target (conservative but active)
        target = expected / 100 * 0.5
        return max(cfg.TARGET_CAP_MIN, min(cfg.TARGET_CAP_MAX, target))

    # Fallback: use quarterly target as default per-stock target
    qcfg = _load_quarterly_config()
    return _get_target_pct(qcfg)


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════════

def _get_live_price(ticker: str) -> float:
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def _get_portfolio_snapshot() -> Dict:
    portfolio = load_portfolio()
    cash = portfolio.get("cash", 0)
    holdings_raw = portfolio.get("holdings", {})

    total_market_value = 0.0
    holding_details = []

    for ticker, data in holdings_raw.items():
        holding = _normalise_holding(data)
        qty = holding.get("qty", 0)
        entry_price = holding.get("entry_price", 0)
        entry_date = holding.get("entry_date", "unknown")
        peak_price = holding.get("peak_price", entry_price)

        if qty <= 0:
            continue

        ltp = _get_live_price(ticker) or entry_price
        new_peak = max(peak_price, ltp) if peak_price > 0 else ltp

        value = qty * ltp
        total_market_value += value
        gain_pct = (ltp - entry_price) / entry_price if entry_price > 0 else 0

        days_held = 0
        if entry_date not in (None, "", "unknown"):
            try:
                days_held = (date.today() - datetime.strptime(entry_date, "%Y-%m-%d").date()).days
            except Exception:
                pass

        holding_details.append({
            "ticker": ticker,
            "qty": qty,
            "entry_price": entry_price,
            "entry_date": entry_date,
            "ltp": ltp,
            "peak_price": new_peak,
            "value": value,
            "gain_pct": gain_pct,
            "days_held": days_held,
        })

    total_value = cash + total_market_value
    return {
        "cash": cash,
        "total_market_value": total_market_value,
        "total_value": total_value,
        "holdings_count": len(holding_details),
        "holdings": holding_details,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING SYSTEM (for proactive rebalancing)
# ═══════════════════════════════════════════════════════════════════════════════

def score_holding(ticker: str, holding: Dict, optimal_params: Dict) -> float:
    """
    Score an existing holding. Lower = weaker candidate for sale.
    Composite score based on: gain/loss, days held, expected_return, stability.
    """
    gain_pct = holding.get("gain_pct", 0)
    days_held = holding.get("days_held", 0)
    plan = optimal_params.get(ticker, {})

    expected = plan.get("expected_return", 0)
    stability = plan.get("stability_score", 0)
    sharpe = plan.get("sharpe_ratio", 0)
    composite = plan.get("composite_score", 0)

    # Base score: reward current gain, penalize stagnation
    score = (
        gain_pct * 100 * 0.40 +           # 40% weight on current performance
        expected * 0.25 +                  # 25% weight on expected return
        stability * 0.20 +                 # 20% weight on stability
        sharpe * 5 * 0.10 +                # 10% weight on Sharpe
        composite * 100 * 0.05              # 5% weight on composite
    )

    # Penalties
    cfg = ActiveConfig
    if days_held > cfg.DEAD_MONEY_DAYS and gain_pct < cfg.DEAD_MONEY_MIN_GAIN:
        score -= 25  # Dead money penalty
    if gain_pct < -0.05:
        score -= 15
    if gain_pct < -0.08:
        score -= 25
    if days_held > 30 and gain_pct < 0:
        score -= 10  # Stagnation penalty

    return score


def score_opportunity(ticker: str, optimal_params: Dict) -> float:
    """
    Score a potential buy opportunity from the full universe.
    Higher = more attractive buy candidate.
    """
    plan = optimal_params.get(ticker, {})
    expected = plan.get("expected_return", 0)
    stability = plan.get("stability_score", 0)
    sharpe = plan.get("sharpe_ratio", 0)
    composite = plan.get("composite_score", 0)
    max_dd = plan.get("max_drawdown", 100)

    # Skip if no valid strategy assigned
    strategy = plan.get("strategy", "NONE")
    if strategy == "NONE" or expected <= 0:
        return -999  # Not a valid opportunity

    # Score: high expected return + high stability + low drawdown
    score = (
        expected * 0.35 +                  # 35% weight on expected return
        stability * 0.25 +                 # 25% weight on stability
        sharpe * 5 * 0.20 +                # 20% weight on Sharpe
        composite * 100 * 0.15 +           # 15% weight on composite
        (100 - max_dd) * 0.05              # 5% weight on drawdown protection
    )

    return score


# ═══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def get_dynamic_risk_multiplier(current_return_pct: float, target_pct: float) -> float:
    """Scale risk based on distance from quarterly target."""
    if current_return_pct >= target_pct * 0.6:
        return ActiveConfig.RISK_DEFENSIVE
    elif current_return_pct <= -0.05:
        return 0.5
    else:
        return ActiveConfig.RISK_NORMAL


def milestone_check(actual_return: float, days_elapsed: int,
                    target_pct: float, quarter_days: int) -> Dict:
    """Are we on track to hit the quarterly target?"""
    expected = (days_elapsed / quarter_days) * target_pct if quarter_days > 0 else 0

    if actual_return >= expected * ActiveConfig.AHEAD_MULTIPLIER:
        action = "LOCK_IN"
        msg = f"Ahead: {actual_return*100:.1f}% vs expected {expected*100:.1f}%"
    elif actual_return < expected * ActiveConfig.BEHIND_MULTIPLIER:
        action = "PUSH"
        msg = f"Behind: {actual_return*100:.1f}% vs expected {expected*100:.1f}%"
    else:
        action = "HOLD"
        msg = f"On track: {actual_return*100:.1f}% vs expected {expected*100:.1f}%"

    return {
        "action": action,
        "message": msg,
        "actual_pct": actual_return * 100,
        "expected_pct": expected * 100,
        "days_elapsed": days_elapsed,
        "days_left": quarter_days - days_elapsed,
        "shortfall_pct": max(0, (target_pct - actual_return) * 100),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVE PROFIT ENGINE v2
# ═══════════════════════════════════════════════════════════════════════════════

class ActiveProfitEngine:
    """
    Enhanced engine with:
      - Per-stock profit targets from optimal_params
      - Proactive rebalancing across full universe
      - Portfolio-level milestone tracking
    """

    def __init__(self):
        self.qcfg = _load_quarterly_config()
        self.optimal = _load_optimal_params()
        self.universe = _load_stocks_universe()
        self.base = _get_base_capital(self.qcfg)
        self.target = _get_target_pct(self.qcfg)
        self.quarter_days = _get_quarter_days(self.qcfg)
        self.quarter_start = _get_quarter_start(self.qcfg)
        self.snapshot = _get_portfolio_snapshot()

    def run(self) -> Dict:
        """
        Execute full active evaluation.
        Returns: result dict with all actions and risk multiplier.
        """
        current_return = (self.snapshot["total_value"] - self.base) / self.base if self.base > 0 else 0
        days_elapsed = (date.today() - self.quarter_start).days

        result = {
            "base_capital": self.base,
            "target_pct": self.target * 100,
            "current_return_pct": current_return * 100,
            "days_elapsed": days_elapsed,
            "quarter_days": self.quarter_days,
            "milestone": milestone_check(current_return, days_elapsed, self.target, self.quarter_days),
            "risk_multiplier": get_dynamic_risk_multiplier(current_return, self.target),
            "trims": [],
            "cuts": [],
            "dead_money": [],
            "rebalances": [],
            "push_analysis": None,
            "per_stock_targets": {},
        }

        # Calculate per-stock targets
        for h in self.snapshot.get("holdings", []):
            ticker = h["ticker"]
            result["per_stock_targets"][ticker] = get_stock_target_pct(ticker, self.optimal)

        # ── Action routing ──────────────────────────────────────
        milestone_action = result["milestone"]["action"]

        if milestone_action == "PUSH":
            # Behind target: be aggressive
            result["cuts"] = self._execute_cuts(aggressive=True)
            result["trims"] = self._execute_trims()
            result["rebalances"] = self._execute_proactive_rebalancing()
            result["push_analysis"] = self._calculate_push(result["milestone"])

        elif milestone_action == "LOCK_IN":
            # Ahead of target: lock in gains
            result["trims"] = self._execute_trims()

        else:
            # On track: normal maintenance
            result["trims"] = self._execute_trims()
            result["cuts"] = self._execute_cuts(aggressive=False)
            result["rebalances"] = self._execute_proactive_rebalancing()

        # Dead money exits run regardless
        result["dead_money"] = self._execute_dead_money()

        return result

    # ── Trims ──────────────────────────────────────────────────

    def _execute_trims(self) -> List[Dict]:
        """Trim top winners to lock in partial profit and free cash."""
        executed = []
        holdings = self.snapshot.get("holdings", [])
        if len(holdings) <= 3:
            return executed

        # Sort by gain % descending
        winners = sorted(holdings, key=lambda x: x["gain_pct"], reverse=True)

        for w in winners[:ActiveConfig.TRIM_MAX_PER_RUN]:
            target_pct = get_stock_target_pct(w["ticker"], self.optimal)

            # Trim if at or above target, or if above threshold
            if w["gain_pct"] < ActiveConfig.TRIM_GAIN_THRESHOLD and w["gain_pct"] < target_pct:
                continue

            trim_qty = max(1, int(w["qty"] * ActiveConfig.TRIM_FRACTION))
            if trim_qty >= w["qty"]:
                trim_qty = max(1, w["qty"] - 1)

            print(f"   🟢 TRIM {w['ticker']}: Sell {trim_qty}/{w['qty']} @ ₹{w['ltp']:.2f} "
                  f"(+{w['gain_pct']*100:.1f}%, target {target_pct*100:.1f}%)")

            record_transaction(
                ticker=w["ticker"],
                side="sell",
                qty=trim_qty,
                price=w["ltp"],
                strategy_name=f"ActiveTrim (+{w['gain_pct']*100:.0f}%, target {target_pct*100:.0f}%)"
            )
            executed.append({
                "ticker": w["ticker"],
                "action": "TRIM",
                "qty": trim_qty,
                "price": w["ltp"],
                "gain_pct": w["gain_pct"],
                "target_pct": target_pct,
            })

        return executed

    # ── Cuts ───────────────────────────────────────────────────

    def _execute_cuts(self, aggressive: bool = False) -> List[Dict]:
        """Cut losers early to stop the bleed."""
        executed = []
        holdings = self.snapshot.get("holdings", [])

        losers = sorted(holdings, key=lambda x: x["gain_pct"])

        max_cuts = ActiveConfig.CUT_MAX_AGGRESSIVE if aggressive else ActiveConfig.CUT_MAX_PER_RUN
        threshold = ActiveConfig.CUT_AGGRESSIVE_THRESHOLD if aggressive else ActiveConfig.CUT_LOSS_THRESHOLD

        for l in losers[:max_cuts]:
            if l["gain_pct"] > threshold:
                continue

            print(f"   🔴 CUT {l['ticker']}: Sell {l['qty']} @ ₹{l['ltp']:.2f} "
                  f"({l['gain_pct']*100:.1f}%)")

            record_transaction(
                ticker=l["ticker"],
                side="sell",
                qty=l["qty"],
                price=l["ltp"],
                strategy_name=f"ActiveCut ({l['gain_pct']*100:.1f}%)"
            )
            executed.append({
                "ticker": l["ticker"],
                "action": "CUT",
                "qty": l["qty"],
                "price": l["ltp"],
                "loss_pct": l["gain_pct"],
            })

        return executed

    # ── Dead Money ─────────────────────────────────────────────

    def _execute_dead_money(self) -> List[Dict]:
        """Exit positions held too long without meaningful progress."""
        executed = []

        for h in self.snapshot.get("holdings", []):
            if h["days_held"] < ActiveConfig.DEAD_MONEY_DAYS:
                continue
            if h["gain_pct"] >= ActiveConfig.DEAD_MONEY_MIN_GAIN:
                continue

            print(f"   ⏳ DEAD MONEY {h['ticker']}: Held {h['days_held']}d, "
                  f"{h['gain_pct']*100:.1f}%. Sell {h['qty']} shares.")

            record_transaction(
                ticker=h["ticker"],
                side="sell",
                qty=h["qty"],
                price=h["ltp"],
                strategy_name=f"DeadMoney ({h['days_held']}d, {h['gain_pct']*100:.1f}%)"
            )
            executed.append({
                "ticker": h["ticker"],
                "action": "DEAD_MONEY",
                "qty": h["qty"],
                "price": h["ltp"],
                "days_held": h["days_held"],
                "gain_pct": h["gain_pct"],
            })

        return executed

    # ── Proactive Rebalancing (NEW v2) ───────────────────────

    def _execute_proactive_rebalancing(self) -> List[Dict]:
        """
        Proactively scan full universe for better opportunities.
        If a non-held stock scores significantly higher than a held stock,
        sell the weak holding and prepare to buy the better opportunity.
        """
        executed = []
        holdings = self.snapshot.get("holdings", [])
        cfg = ActiveConfig

        if len(holdings) < cfg.REBALANCE_MAX_POSITIONS:
            return executed  # Room to add without selling

        # Score all current holdings
        holding_scores = []
        for h in holdings:
            score = score_holding(h["ticker"], h, self.optimal)
            holding_scores.append((score, h))

        # Score all universe opportunities (not currently held)
        held_tickers = {h["ticker"] for h in holdings}
        opportunity_scores = []
        for ticker in self.universe:
            if ticker in held_tickers:
                continue
            score = score_opportunity(ticker, self.optimal)
            if score > -500:  # Valid opportunity
                opportunity_scores.append((score, ticker))

        if not opportunity_scores:
            return executed

        # Sort: weakest holdings first, best opportunities first
        holding_scores.sort(key=lambda x: x[0])
        opportunity_scores.sort(key=lambda x: x[0], reverse=True)

        # Check if we should rotate
        for opp_score, opp_ticker in opportunity_scores[:3]:  # Top 3 opportunities
            for hold_score, holding in holding_scores[:2]:  # Bottom 2 holdings
                if opp_score >= hold_score + cfg.REBALANCE_MIN_GAP:
                    # Sell weak holding to free cash for better opportunity
                    print(f"   🔄 REBALANCE: {holding['ticker']} (score {hold_score:.1f}) → "
                          f"{opp_ticker} (score {opp_score:.1f})")

                    record_transaction(
                        ticker=holding["ticker"],
                        side="sell",
                        qty=holding["qty"],
                        price=holding["ltp"],
                        strategy_name=f"Rebalance → {opp_ticker}"
                    )
                    executed.append({
                        "ticker": holding["ticker"],
                        "action": "REBALANCE_SELL",
                        "qty": holding["qty"],
                        "price": holding["ltp"],
                        "score": hold_score,
                        "replacement": opp_ticker,
                        "replacement_score": opp_score,
                    })
                    break  # Only one rebalance per opportunity

        return executed

    # ── Push Analysis ──────────────────────────────────────────

    def _calculate_push(self, milestone: Dict) -> Dict:
        """When behind target, calculate catch-up metrics."""
        shortfall = milestone["shortfall_pct"]
        days_left = milestone["days_left"]

        if days_left <= 0:
            return {"shortfall_pct": shortfall, "days_left": 0,
                    "required_daily": 0, "note": "Quarter ended"}

        required_daily = shortfall / days_left

        return {
            "shortfall_pct": shortfall,
            "days_left": days_left,
            "required_daily_pct": required_daily,
            "note": (f"Need {shortfall:.2f}% more in {days_left}d "
                     f"({required_daily:.3f}% per day). Risk multiplier: {ActiveConfig.RISK_AGGRESSIVE}x")
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    """Execute active profit engine and print report."""
    print("\n" + "=" * 60)
    print("🎯 T_RAIDER ACTIVE PROFIT ENGINE v2")
    print("=" * 60)

    engine = ActiveProfitEngine()
    result = engine.run()

    m = result["milestone"]
    print(f"\n📊 Quarter Status")
    print(f"   Base Capital: ₹{result['base_capital']:,.0f}")
    print(f"   Target: +{result['target_pct']:.1f}%")
    print(f"   Current Return: {result['current_return_pct']:+.2f}%")
    print(f"   Days: {result['days_elapsed']}/{result['quarter_days']}")
    print(f"   Milestone: {m['action']} — {m['message']}")
    print(f"   Risk Multiplier: {result['risk_multiplier']:.1f}x")

    if result["per_stock_targets"]:
        print(f"\n📋 Per-Stock Targets (sample):")
        for ticker, target in list(result["per_stock_targets"].items())[:5]:
            print(f"   {ticker}: {target*100:.1f}%")

    if result["trims"]:
        print(f"\n🟢 Trims ({len(result['trims'])}):")
        for t in result["trims"]:
            print(f"   {t['ticker']}: {t['qty']} @ ₹{t['price']:.2f} ({t['gain_pct']*100:.1f}%)")

    if result["cuts"]:
        print(f"\n🔴 Cuts ({len(result['cuts'])}):")
        for c in result["cuts"]:
            print(f"   {c['ticker']}: {c['qty']} @ ₹{c['price']:.2f} ({c['loss_pct']*100:.1f}%)")

    if result["dead_money"]:
        print(f"\n⏳ Dead Money ({len(result['dead_money'])}):")
        for d in result["dead_money"]:
            print(f"   {d['ticker']}: {d['qty']} @ ₹{d['price']:.2f} ({d['days_held']}d)")

    if result["rebalances"]:
        print(f"\n🔄 Rebalances ({len(result['rebalances'])}):")
        for r in result["rebalances"]:
            print(f"   {r['ticker']} → {r['replacement']} ({r['score']:.1f} vs {r['replacement_score']:.1f})")

    if result["push_analysis"]:
        p = result["push_analysis"]
        print(f"\n⚡ Push Analysis:")
        print(f"   {p['note']}")

    print("\n" + "=" * 60)
    return result


if __name__ == "__main__":
    run()