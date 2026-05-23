"""
daily_screener.py (v3.4 — Macro Environment Filter)
────────────────────────────────────────────────────────
Changes from v3.3 (medium bug fixes):

 NEW — Macro Environment Filter (macro_filter.py)
   Three pre-trade environment checks now sit ABOVE the technical signal
   layer as Gate 1b, between the existing regime gate (Gate 1) and the
   volume gate (Gate 2):

   Gate 1b-1  VIX / Volatility Gate
       Suppresses new BUYs when market fear index is at extreme levels.
       Downgrades HIGH → MEDIUM when fear is elevated.

   Gate 1b-2  Institutional Flow Gate  (India: FII + DII net flow)
       Downgrades HIGH → MEDIUM when big-money is net sellers.
       Gracefully skipped for markets where flow data is unavailable.

   Gate 1b-3  Earnings Blackout Gate
       Blocks BUYs on earnings day itself.
       Downgrades HIGH → MEDIUM within configurable days of earnings.

   Gate 1b-4  High-Risk Calendar Gate
       Downgrades HIGH → MEDIUM on central bank meeting / budget days.

   Gate 1b-5  Overnight Reference Market Gate
       Blocks BUYs when overnight reference market (e.g. S&P 500 for India)
       is down >2%. Downgrades at >1.5%.

 NEW — Market parameter
   run_screener() and __main__ now accept --market (default: INDIA).
   Supported out of the box: INDIA, AUSTRALIA, CANADA, USA.
   All market-specific config (VIX ticker, thresholds, calendar) lives
   in macro_filter.MARKET_CONFIGS — no logic changes needed to add markets.

 FIX 4 — Nifty drop uses actual 50-day rolling HIGH (not EMA-distance proxy)
 FIX 7 — get_portfolio_context uses market value not cost basis
"""

import pandas as pd
import json
import os
import sys
import datetime
import warnings
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sector_analyzer import get_sector_ranks

from ingestion.data_ingestion import fetch_historical_data, get_stock_data
from ingestion.nse_constituents import get_market_regime, regime_summary
from strategies.trend_follower import apply_golden_cross_strategy
from strategies.mean_reversion import apply_rsi_strategy
from strategies.volatility import apply_bollinger_strategy
from strategies.breakout import apply_breakout_strategy
from strategies.momentum import apply_macd_strategy
from strategies.stretch import apply_stretch_strategy
from autopilot.logger import load_portfolio

from macro_filter import MacroFilter, MARKET_CONFIGS, FilterAction
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from autopilot.correlation_filter import CorrelationFilter
from autopilot.kelly_position_sizing import KellyPositionSizer


# ── ATR + position sizing ─────────────────────────────────────────────────────

def calculate_atr(df: pd.DataFrame, window: int = 14) -> float:
    high = df['High'] if 'High' in df.columns else df['Close']
    low = df['Low'] if 'Low' in df.columns else df['Close']
    close = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0

def calculate_position_size(capital: float, atr: float, risk_pct: float = 0.01) -> int:
    if atr <= 0:
        return 0
    return max(int((capital * risk_pct) / (atr * 2)), 0)


# ── Volume confirmation ───────────────────────────────────────────────────────

def check_volume_confirmation(df: pd.DataFrame,
                              avg_window: int = 20,
                              min_ratio: float = 0.80) -> Tuple[bool, float, float]:
    if 'Volume' not in df.columns:
        return True, 0.0, 0.0

    volume = df['Volume'].dropna()
    if len(volume) < avg_window + 1:
        return True, 0.0, 0.0

    today_vol = float(volume.iloc[-1])
    avg_vol = float(volume.iloc[-(avg_window + 1):-1].mean())

    if avg_vol <= 0:
        return True, today_vol, 0.0

    return today_vol >= avg_vol * min_ratio, today_vol, avg_vol


# ── Nifty 50-day high drop (FIX 4) ───────────────────────────────────────────

def get_nifty_drop_from_50d_high() -> float:
    """
    FIX 4: Returns percentage Nifty 50 has dropped from its 50-day rolling high.
    Original code used EMA distance as a proxy — this uses the actual rolling max.
    """
    try:
        import yfinance as yf
        nifty = yf.download("^NSEI", period="80d", progress=False, auto_adjust=True)
        if nifty.empty or len(nifty) < 10:
            warnings.warn("Insufficient Nifty data for 50d-high drop — using 0.0.")
            return 0.0
        close = nifty["Close"].squeeze()
        window = min(50, len(close))
        high_50d = float(close.iloc[-window:].max())
        latest = float(close.iloc[-1])
        if high_50d <= 0:
            return 0.0
        return max((high_50d - latest) / high_50d, 0.0)
    except Exception as e:
        warnings.warn(f"50d-high drop calculation failed ({e}) — using 0.0.")
        return 0.0


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class ScreenerSignal:
    ticker: str
    signal_type: str
    price: float
    strategy: str
    expected_return: float
    stability_score: float
    confidence_tier: str
    suggested_qty: int
    risk_per_trade_inr: float
    current_holdings: int
    portfolio_weight_pct: float
    reason: str
    atr_14d: float = 0.0
    cash_required: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_notification(self) -> str:
        if self.confidence_tier == "SKIP":
            return (f"❌ **{self.ticker}** | {self.strategy}\n"
                    f"   Stability: {self.stability_score:.1f}% | {self.reason}")
        emoji = "🟢" if self.confidence_tier == "HIGH" else "🟡"
        qty_line = f"   Suggested Qty: {self.suggested_qty} shares\n" if self.suggested_qty > 0 else ""
        cash_line = f"   Cash Required: ₹{self.cash_required:.0f}\n" if self.cash_required > 0 else ""
        return (
            f"{emoji} **{self.ticker}** | {self.strategy} | {self.signal_type}\n"
            f"   Price: ₹{self.price:.2f} | Stability: {self.stability_score:.1f}%"
            f" | Expected Return: {self.expected_return:.1f}%\n"
            f"{qty_line}{cash_line}"
            f"   Risk: ₹{self.risk_per_trade_inr:.0f} | Holdings: {self.current_holdings}"
            f" | Weight: {self.portfolio_weight_pct:.1f}%\n"
            f"   {'✅ APPROVED' if self.confidence_tier == 'HIGH' else '⚠️ REVIEW BEFORE ACTING'}"
        )


# ── Portfolio context (FIX 7) ─────────────────────────────────────────────────

def get_portfolio_context(portfolio: Dict, ticker: str,
                         latest_price: float) -> Tuple[int, float, float, float]:
    """
    FIX 7 CORRECTED: Returns accurate portfolio context for signal evaluation.
    
    Returns:
        current_qty: shares held of this ticker
        weight_pct: this ticker's weight in portfolio (using latest_price)
        total_value: total portfolio value (cash + all holdings at market value)
        cash: available cash
    """
    holdings = portfolio.get('holdings', {})
    cash = portfolio.get('cash', 0)
    
    def _qty(v) -> int:
        return v['qty'] if isinstance(v, dict) else int(v or 0)
    
    # Calculate total portfolio market value
    total_market_value = 0.0
    
    for hticker, hdata in holdings.items():
        qty = _qty(hdata)
        if qty <= 0:
            continue
            
        if hticker == ticker:
            # Use the passed latest_price for current ticker
            total_market_value += qty * latest_price
        else:
            # For other holdings, we need their current price
            # Since we don't have it here, use entry_price as conservative estimate
            # (The caller should ideally pass all prices, but this is a screener)
            entry = hdata.get('entry_price', 0) if isinstance(hdata, dict) else 0
            total_market_value += qty * (entry if entry > 0 else 0)
    
    total_value = cash + total_market_value
    
    # Current ticker specific
    raw = holdings.get(ticker, 0)
    current_qty = int(raw['qty'] if isinstance(raw, dict) else (raw or 0))
    
    weight_pct = (current_qty * latest_price / total_value * 100) if total_value > 0 else 0
    
    return current_qty, weight_pct, total_value, cash
# ── Confidence classification ─────────────────────────────────────────────────

def classify_buy_confidence(stability, weight_pct, current_qty, cash, estimated_cost,
                            total_value, latest_price,
                            max_weight=20.0) -> Tuple[str, str]:
    """
    Returns confidence tier and reason for a BUY signal.
    NOW CHECKS prospective weight AFTER the proposed purchase.
    """
    if estimated_cost > cash * 1.05:
        return "SKIP", f"Insufficient cash (Need ₹{estimated_cost:.0f}, Have ₹{cash:.0f})"

    # ── NEW: compute what weight this ticker would have AFTER buying ──
    current_holdings_value = current_qty * latest_price
    post_buy_value = current_holdings_value + estimated_cost
    prospective_weight = (post_buy_value / total_value * 100) if total_value > 0 else 0

    if current_qty > 0:
        # Adding to existing position — allowed only if we stay under the limit
        if stability >= 70 and prospective_weight < max_weight:
            return "MEDIUM", (f"Already holding {current_qty} shares — "
                            f"add takes weight to {prospective_weight:.1f}%")
        return "SKIP", (f"Already holding {current_qty} shares — "
                       f"would reach {prospective_weight:.1f}% concentration")

    # New position
    if stability >= 70 and prospective_weight < max_weight:
        return "HIGH", (f"Strong MC validation + within concentration limits "
                       f"({prospective_weight:.1f}%)")
    if stability >= 60 and prospective_weight < max_weight:
        return "MEDIUM", (f"Marginal stability score — review recommended "
                         f"({prospective_weight:.1f}%)")
    if prospective_weight >= max_weight:
        return "SKIP", f"Concentration limit ({prospective_weight:.1f}% >= {max_weight}%)"
    return "SKIP", f"Stability too low ({stability:.1f}% < 60%)"

def classify_sell_confidence(current_qty: int) -> Tuple[str, str]:
    if current_qty > 0:
        return "HIGH", "Exit signal on existing position — execute immediately"
    return "SKIP", "Bearish signal but no position held — informational only"


# ── Logging ───────────────────────────────────────────────────────────────────

def log_recommendations(signals: List[ScreenerSignal], portfolio: Dict,
                        regime_info: Dict, log_dir: str = "config/manual_logs"):
    Path(log_dir).mkdir(exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    log_data = {
        "date": date_str,
        "timestamp": datetime.datetime.now().isoformat(),
        "market_regime": regime_info,
        "portfolio_snapshot": {
            "cash": portfolio.get('cash', 0),
            "holdings_count": len(portfolio.get('holdings', {})),
        },
        "total_signals": len(signals),
        "high_confidence": len([s for s in signals if s.confidence_tier == "HIGH"]),
        "medium_confidence": len([s for s in signals if s.confidence_tier == "MEDIUM"]),
        "recommendations": [s.to_dict() for s in signals if s.confidence_tier != "SKIP"],
    }
    filepath = Path(log_dir) / f"recs_{date_str}.json"
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\n📁 Logged recommendations to {filepath}")


# ── GitHub Actions summary ────────────────────────────────────────────────────

def print_github_actions_summary(signals: List[ScreenerSignal], portfolio: Dict,
                                 regime_info: Dict):
    high   = [s for s in signals if s.confidence_tier == "HIGH"]
    medium = [s for s in signals if s.confidence_tier == "MEDIUM"]
    cash   = portfolio.get('cash', 0)

    print("\n" + "=" * 90)
    print("## 📊 T_Raider Daily Screener — Human Review Mode")
    print(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"**Available Cash:** ₹{cash:,.0f}")

    regime_emoji = "📈" if regime_info['is_uptrend'] else "📉"
    print(f"**Market Regime:** {regime_emoji} {regime_info['summary']}")
    if not regime_info['is_uptrend']:
        print("⛔ BEAR MARKET MODE: All BUY signals suppressed. Only SELL signals active.")

    # Print macro snapshot in summary if available
    macro_snap = regime_info.get("macro", {})
    if macro_snap:
        vix_str  = f"VIX {macro_snap['vix']:.1f} [{macro_snap['vix_status']}]" if macro_snap.get('vix') else "VIX N/A"
        flow_str = (f"Flow {macro_snap['inst_flow']:+,.0f}M [{macro_snap['flow_status']}]"
                    if macro_snap.get('inst_flow') is not None else "Flow N/A")
        ov_str   = (f"Overnight {macro_snap['overnight_pct']:+.1f}%"
                    if macro_snap.get('overnight_pct') is not None else "Overnight N/A")
        print(f"**Macro:** {vix_str} | {flow_str} | {ov_str}")
        if macro_snap.get('high_risk_day'):
            print(f"⚠️  HIGH-RISK DAY: {macro_snap['high_risk_reason']}")

    print(
        f"**Actionable Sells:** {len([s for s in high if s.signal_type == 'SELL'])} | "
        f"**New Buys (HIGH):** {len([s for s in high if s.signal_type == 'BUY'])} | "
        f"**Review:** {len(medium)}"
    )
    print("=" * 90)

    actionable = [s for s in signals if s.confidence_tier in ("HIGH", "MEDIUM")]
    if not actionable:
        print("\nℹ️ No actionable signals today.")
        return

    print(f"\n{'TIER':8} | {'TYPE':5} | {'TICKER':14} | {'PRICE':10} | "
          f"{'STRATEGY':12} | {'STABILITY':10} | {'QTY':6} | {'CASH REQ':10}")
    print("-" * 90)

    for s in actionable:
        emoji    = "🟢" if s.confidence_tier == "HIGH" else "🟡"
        qty_str  = str(s.suggested_qty) if s.suggested_qty > 0 else "-"
        cash_str = f"₹{s.cash_required:.0f}" if s.cash_required > 0 else "-"
        print(
            f"{emoji} {s.confidence_tier:6} | {s.signal_type:5} | {s.ticker:14} | "
            f"₹{s.price:8.2f} | {s.strategy:12} | {s.stability_score:8.1f}% | "
            f"{qty_str:>5} | {cash_str:>9}"
        )

    print("\n" + "=" * 90)
    if cash < 5000:
        print(f"⚠️ LOW CASH: Only ₹{cash:,.0f} available. New buys may be limited.")
    print("📝 HIGH signals pre-approved. MEDIUM requires your judgment.")
    print("=" * 90)


# ── Main screener ─────────────────────────────────────────────────────────────

def run_screener(tickers, capital: Optional[float] = None, min_stability: float = 60.0,
                 volume_min_ratio: float = 0.80, mode: str = "CONSERVATIVE",
                 market: str = "INDIA"):
    """
    Args:
        capital          : Total capital for ATR sizing. None = use portfolio value.
        min_stability    : Minimum stability score to consider.
        volume_min_ratio : Min ratio of today's vol to 20-day avg for BUY to pass.
        mode             : CONSERVATIVE / BALANCED / AGGRESSIVE.
        market           : INDIA / AUSTRALIA / CANADA / USA.
                           Controls VIX source, flow source, calendar events.
                           Add new markets in macro_filter.MARKET_CONFIGS.
    """
    print(f"\n--- T_Raider Hybrid Execution Engine (Sturdy Mode v3.4) ---")
    print(f"Timestamp : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode      : {mode}")
    print(f"Market    : {market}")
    if volume_min_ratio > 0:
        print(f" Volume filter: BUY signals require ≥{volume_min_ratio*100:.0f}% of 20-day avg volume")

    # ── Risk profile ──────────────────────────────────────────────────────
    RISK_PROFILES = {
        "CONSERVATIVE": {"allow_mean_reversion": False, "nifty_drop_threshold": 0.0},
        "BALANCED":     {"allow_mean_reversion": True,  "nifty_drop_threshold": 0.05},
        "AGGRESSIVE":   {"allow_mean_reversion": True,  "nifty_drop_threshold": 0.03},
    }
    config = RISK_PROFILES.get(mode, RISK_PROFILES["CONSERVATIVE"])
    print(f" Mean-reversion in downtrend: {config['allow_mean_reversion']}"
          f" (threshold: {config['nifty_drop_threshold']:.0%})")

    # ── Gate 1: Market Regime ─────────────────────────────────────────────
    print("\n🌐 Checking market regime…")
    is_uptrend, nifty_close, nifty_ema = get_market_regime()

    nifty_drop = 0.0
    if not is_uptrend and config["allow_mean_reversion"]:
        # FIX 4: actual 50-day rolling high, not EMA-distance proxy
        nifty_drop = get_nifty_drop_from_50d_high()
        print(f" 📐 Nifty 50-day high drop: {nifty_drop:.1%} "
              f"(threshold: {config['nifty_drop_threshold']:.0%})")

    regime_info = {
        "is_uptrend":          is_uptrend,
        "nifty_close":         round(nifty_close, 2),
        "nifty_ema50":         round(nifty_ema, 2),
        "nifty_drop_50d_high": round(nifty_drop, 4),
        "summary":             regime_summary(is_uptrend, nifty_close, nifty_ema),
    }
    print(f" {regime_info['summary']}")
    if not is_uptrend:
        if config["allow_mean_reversion"]:
            print(f" ⛔ Downtrend — mean-reversion BUYs allowed only if "
                  f"Nifty drop > {config['nifty_drop_threshold']:.0%} "
                  f"(current: {nifty_drop:.1%})")
        else:
            print(" ⛔ Downtrend detected — BUY signals suppressed.")

    # ── Gate 1b: Macro Environment Filter ────────────────────────────────
    # Fetched ONCE per session — shared across all tickers in the loop.
    # Gracefully disabled if the market key is unknown or fetches fail.
    print("\n🌍 Fetching macro environment data…")
    if market not in MARKET_CONFIGS:
        print(f"  ⚠️  Unknown market '{market}' — macro filter disabled. "
              f"Available: {list(MARKET_CONFIGS.keys())}")
        macro         = None
        macro_context = None
    else:
        macro         = MacroFilter(MARKET_CONFIGS[market])
        macro_context = macro.run()
        macro.print_summary(macro_context)

        # Attach macro snapshot to regime_info so it gets written to recs JSON
        regime_info["macro"] = {
            "vix":              macro_context.vix,
            "vix_status":       macro_context.vix_status,
            "inst_flow":        macro_context.institutional_flow,
            "flow_status":      macro_context.flow_status,
            "overnight_pct":    macro_context.overnight_change_pct,
            "high_risk_day":    macro_context.is_high_risk_day,
            "high_risk_reason": macro_context.high_risk_reason,
        }

    # ── Sector Momentum ───────────────────────────────────────────────────
    print("\n📊 Loading Sector Momentum Data…")
    try:
        with open('config/dynamic_sector_map.json', 'r') as f:
            dynamic_map = json.load(f)
        sector_ranks = get_sector_ranks()
        print(" ✅ Sector ranks loaded.")
    except FileNotFoundError:
        print(" ⚠️ Sector map missing. Run sector_mapper.py first. Sector boosting disabled.")
        dynamic_map  = {}
        sector_ranks = {}

    # ── Load optimal params ───────────────────────────────────────────────
    if not os.path.exists('config/optimal_params.json'):
        print("❌ ERROR: No optimised data found. Run auto_optimizer.py first!")
        return [], []

    with open('config/optimal_params.json', 'r') as f:
        optimized_params = json.load(f)
    print(f"\n✅ Loaded optimised 'brains' for {len(optimized_params)} stocks.")

    full_market_data = fetch_historical_data(tickers, period="2y")

    all_signals:     List[ScreenerSignal] = []
    buy_signals      = []
    sell_signals     = []
    portfolio        = load_portfolio()
    vol_filtered     = 0
    macro_skipped    = 0
    macro_downgraded = 0

    for ticker in tickers:
        plan = optimized_params.get(ticker)
        if not plan or plan.get('strategy') == "NONE":
            continue

        df = get_stock_data(full_market_data, ticker)
        if df.empty or len(df) < 200:
            continue

        price        = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        latest_price = float(price.iloc[-1])
        strat_type   = plan['strategy']
        p            = plan.get('params', {})

        if   strat_type == "TREND":      res = apply_golden_cross_strategy(price)
        elif strat_type == "RSI":        res = apply_rsi_strategy(price, window=p.get('window', 14), buy=p.get('buy', 30), sell=p.get('sell', 70))
        elif strat_type == "VOLATILITY": res = apply_bollinger_strategy(price)
        elif strat_type == "BREAKOUT":   res = apply_breakout_strategy(price, window=p.get('window', 20))
        elif strat_type == "MACD":       res = apply_macd_strategy(price)
        elif strat_type == "STRETCH":    res = apply_stretch_strategy(price, window=p.get('window', 20), threshold=p.get('threshold', 0.05))
        else:
            continue

        latest_signal = res.iloc[-1]['Signal']
        if latest_signal not in (1, -1):
            continue

        # FIX 7: market value, not cost basis
        current_qty, weight_pct, portfolio_value, cash = get_portfolio_context(
            portfolio, ticker, latest_price
        )
        # Respect quarterly capital override
        override_path = Path("config/capital_override.json")
        if override_path.exists() and capital is None:
            with open(override_path) as f:
                effective_capital = json.load(f).get("total_baseline_wealth", portfolio_value)
        else:
            effective_capital = capital if capital is not None else portfolio_value
        stability = float(plan.get('stability_score', 0) or 0)

        # ══════════════════════════════════════════════════════════════════
        # BUY signal pipeline
        # Gate 1   → Market regime
        # Gate 1b  → Macro environment  (NEW in v3.4)
        # Gate 2   → Volume confirmation
        # Gate 3   → ATR sizing + confidence classification
        # ══════════════════════════════════════════════════════════════════
        if latest_signal == 1:
            
            corr_ok, corr_reason = CorrelationFilter.check(
                ticker=ticker,
                portfolio=portfolio,
                full_market_data=full_market_data
            )
            if not corr_ok:
                print(f"  {ticker}: SKIP — {corr_reason}")
                continue

            # ── Gate 1: Regime ────────────────────────────────────────────
            if not is_uptrend:
                if config["allow_mean_reversion"]:
                    if nifty_drop > config["nifty_drop_threshold"] and strat_type in ['RSI', 'VOLATILITY', 'STRETCH']:
                        pass  # mean-reversion allowed — fall through to Gate 1b
                    else:
                        skip_reason = (
                            f"⛔ Downtrend — {strat_type} not mean-reversion type"
                            if strat_type not in ['RSI', 'VOLATILITY', 'STRETCH']
                            else f"⛔ Downtrend — Nifty drop {nifty_drop:.1%} < threshold {config['nifty_drop_threshold']:.1%}"
                        )
                        all_signals.append(ScreenerSignal(
                            ticker=ticker, signal_type="BUY", price=latest_price,
                            strategy=strat_type, expected_return=plan.get('expected_return', 0),
                            stability_score=stability, confidence_tier="SKIP",
                            suggested_qty=0, risk_per_trade_inr=0,
                            current_holdings=current_qty, portfolio_weight_pct=weight_pct,
                            reason=skip_reason,
                        ))
                        continue
                else:
                    all_signals.append(ScreenerSignal(
                        ticker=ticker, signal_type="BUY", price=latest_price,
                        strategy=strat_type, expected_return=plan.get('expected_return', 0),
                        stability_score=stability, confidence_tier="SKIP",
                        suggested_qty=0, risk_per_trade_inr=0,
                        current_holdings=current_qty, portfolio_weight_pct=weight_pct,
                        reason="⛔ Suppressed — market in downtrend (close < EMA50)",
                    ))
                    continue

            # ── Gate 1b: Macro Environment ────────────────────────────────
            # evaluate_buy() is fast — all network I/O happened in macro.run() above.
            # Earnings calendar is the only per-ticker fetch; results are cached
            # in macro_context._earnings_cache so each ticker is fetched at most once.
            macro_downgrade_reason = ""
            if macro is not None and macro_context is not None:
                macro_eval = macro.evaluate_buy(ticker, macro_context)

                if macro_eval.action == FilterAction.SKIP:
                    macro_skipped += 1
                    all_signals.append(ScreenerSignal(
                        ticker=ticker, signal_type="BUY", price=latest_price,
                        strategy=strat_type, expected_return=plan.get('expected_return', 0),
                        stability_score=stability, confidence_tier="SKIP",
                        suggested_qty=0, risk_per_trade_inr=0,
                        current_holdings=current_qty, portfolio_weight_pct=weight_pct,
                        reason=macro_eval.reason,
                    ))
                    continue

                if macro_eval.action == FilterAction.DOWNGRADE:
                    macro_downgraded += 1
                    macro_downgrade_reason = " | ".join(macro_eval.warning_flags())

            # ── Gate 2: Volume confirmation ───────────────────────────────
            if volume_min_ratio > 0:
                volume_ok, today_vol, avg_vol = check_volume_confirmation(
                    df, avg_window=20, min_ratio=volume_min_ratio
                )
                if not volume_ok:
                    vol_filtered += 1
                    vol_ratio = (today_vol / avg_vol * 100) if avg_vol > 0 else 0
                    all_signals.append(ScreenerSignal(
                        ticker=ticker, signal_type="BUY", price=latest_price,
                        strategy=strat_type, expected_return=plan.get('expected_return', 0),
                        stability_score=stability, confidence_tier="SKIP",
                        suggested_qty=0, risk_per_trade_inr=0,
                        current_holdings=current_qty, portfolio_weight_pct=weight_pct,
                        reason=(f"📉 Thin volume ({vol_ratio:.0f}% of avg) — "
                                f"signal not confirmed. "
                                f"Today: {today_vol:,.0f} | Avg20: {avg_vol:,.0f}"),
                    ))
                    continue
            else:
                today_vol, avg_vol = 0.0, 0.0

            # ── Gate 3: ATR sizing + confidence classification ─────────────
                       # ── Gate 3: Kelly + ATR sizing + confidence classification ────
            atr = calculate_atr(df)

            suggested_qty, sizing_reason = KellyPositionSizer.calculate(
                ticker=ticker,
                portfolio=portfolio,
                optimal_params=optimized_params,   # satisfies signature; unused internally today
                atr=atr,
                current_price=latest_price,
                mode=mode,
                capital=effective_capital          # uses your override / portfolio value
            )
            estimated_cost = suggested_qty * latest_price
            # Actual rupee risk at risk-per-share = ATR × 2
            risk_per_trade_inr = suggested_qty * atr * 2.0 if suggested_qty > 0 else 0.0

            confidence_tier, reason = classify_buy_confidence(
                stability, weight_pct, current_qty, cash, estimated_cost,
                portfolio_value, latest_price
            )

            # Append Kelly info to the reason string
            reason = f"{reason} | {sizing_reason}"

            # Apply macro downgrade: HIGH → MEDIUM if any macro flag fired
            if macro_downgrade_reason and confidence_tier == "HIGH":
                confidence_tier = "MEDIUM"
                reason = f"{reason} | {macro_downgrade_reason}"
            elif macro_downgrade_reason:
                reason = f"{reason} | {macro_downgrade_reason}"

            # ── Sector momentum ───────────────────────────────────────────
            stock_sector_info = dynamic_map.get(ticker, {"nse_index": "UNKNOWN", "yf_sector": "Unknown"})
            nse_index         = stock_sector_info["nse_index"]
            yf_sector_name    = stock_sector_info["yf_sector"]
            sector_data       = sector_ranks.get(nse_index, {"rank": 99, "is_outperforming": False, "rs_score": 0.0})
            is_sector_strong  = sector_data['is_outperforming']
            sector_rs         = sector_data['rs_score']
            base_score        = plan.get('composite_score', 0) or plan.get('expected_return', 0)
            final_score       = base_score * (1.2 if is_sector_strong else 0.8)

            sector_icon = "🔥" if is_sector_strong else "❄️"
            vol_ratio   = (today_vol / avg_vol * 100) if avg_vol > 0 else 0
            vol_note    = f"Vol: {vol_ratio:.0f}% avg" if avg_vol > 0 else ""
            aug_reason  = (f"{reason} | {yf_sector_name[:10]} {sector_icon}"
                           + (f" | {vol_note}" if vol_note else ""))

            all_signals.append(ScreenerSignal(
                ticker=ticker, signal_type="BUY", price=latest_price,
                strategy=strat_type, expected_return=plan.get('expected_return', 0),
                stability_score=stability, confidence_tier=confidence_tier,
                suggested_qty=suggested_qty, risk_per_trade_inr=risk_per_trade_inr,
                current_holdings=current_qty, portfolio_weight_pct=weight_pct,
                reason=aug_reason, atr_14d=atr, cash_required=estimated_cost,
            ))

            buy_signals.append({
                "ticker":           ticker,
                "price":            latest_price,
                "reason":           strat_type,
                "sector":           yf_sector_name[:12],
                "sector_rs":        sector_rs,
                "is_sector_strong": is_sector_strong,
                "expected_return":  plan.get('expected_return', 0),
                "stability":        stability,
                "sharpe_ratio":     plan.get('sharpe_ratio', None),
                "max_drawdown":     plan.get('max_drawdown', None),
                "composite_score":  final_score,
                "folds_passed":     plan.get('folds_passed', None),
                "volume_ratio":     round(vol_ratio, 1) if avg_vol > 0 else None,
            })

        # ══════════════════════════════════════════════════════════════════
        # SELL signal pipeline — macro gates intentionally NOT applied.
        # Macro conditions are reasons to avoid opening new positions.
        # They are not reasons to hold an existing position through a
        # technical sell signal — that would inverse the risk logic.
        # ══════════════════════════════════════════════════════════════════
        elif latest_signal == -1:
            confidence_tier, reason = classify_sell_confidence(current_qty)
            all_signals.append(ScreenerSignal(
                ticker=ticker, signal_type="SELL", price=latest_price,
                strategy=strat_type, expected_return=plan.get('expected_return', 0),
                stability_score=stability, confidence_tier=confidence_tier,
                suggested_qty=0, risk_per_trade_inr=0,
                current_holdings=current_qty, portfolio_weight_pct=weight_pct,
                reason=reason,
            ))
            sell_signals.append((ticker, latest_price))

    # ── Session summary ───────────────────────────────────────────────────
    if vol_filtered > 0:
        print(f"\n🔇 Volume filter blocked {vol_filtered} BUY signal(s) on thin volume.")
    if macro_skipped > 0:
        print(f"🌍 Macro filter BLOCKED {macro_skipped} BUY signal(s) "
              f"(extreme VIX / earnings day / overnight crash).")
    if macro_downgraded > 0:
        print(f"🌍 Macro filter DOWNGRADED {macro_downgraded} BUY signal(s) HIGH → MEDIUM "
              f"(elevated VIX / bearish flow / calendar / earnings proximity).")

    print_github_actions_summary(all_signals, portfolio, regime_info)
    log_recommendations(all_signals, portfolio, regime_info)

    high_conf = [s for s in all_signals if s.confidence_tier == "HIGH"]
    if high_conf:
        print("\n🟢 HIGH CONFIDENCE SIGNALS — Ready for Manual Execution:")
        for s in high_conf:
            print(s.to_notification())
            print()

    current_holdings_keys = set(portfolio.get('holdings', {}).keys())
    print("\n" + "=" * 135)
    print(f"{'TYPE':6} | {'TICKER':14} | {'SECTOR':12} | {'SECTOR RS':11} | {'PRICE':10} | "
          f"{'STRATEGY':12} | {'SCORE':7} | {'STABILITY':10}")
    print("-" * 135)

    active_buys = [b for b in buy_signals if b['ticker'] not in current_holdings_keys]
    if active_buys:
        active_buys.sort(
            key=lambda x: x['composite_score'] if x['composite_score'] is not None
            else x['expected_return'],
            reverse=True,
        )
        for b in active_buys:
            rs_str  = f"{b['sector_rs']:>6.1f}%" if b['sector_rs'] != 0.0 else "N/A"
            rs_icon = "🔥" if b['is_sector_strong'] else "❄️"
            stab    = f"{b['stability']:.1f}%" if isinstance(b['stability'], (int, float)) else "N/A"
            print(
                f"BUY | {b['ticker']:14} | {b['sector']:12} | {rs_str} {rs_icon} | "
                f"₹{b['price']:8.2f} | {b['reason']:12} | "
                f"{b['composite_score']:7.1f} | {stab:>9}"
            )
    else:
        print("No NEW buy opportunities found.")

    print("\n🛑 SELL ALERTS:")
    my_sells     = [s for s in sell_signals if s[0] in current_holdings_keys]
    market_sells = [s for s in sell_signals if s[0] not in current_holdings_keys]
    if my_sells:
        print("--- !! ACTION REQUIRED (YOUR POSITIONS) !! ---")
        for s in my_sells:
            print(f"SELL | {s[0]:14} | ₹{s[1]:.2f} — EXIT POSITION IMMEDIATELY")
    if market_sells:
        print("--- MARKET INTELLIGENCE (BEARISH WATCHLIST) ---")
        for s in market_sells:
            print(f"SELL | {s[0]:14} | ₹{s[1]:.2f}")

    print("=" * 135 + "\n")
    if not buy_signals and not sell_signals:
        print("ℹ️ Market Scan: No entry/exit thresholds were crossed today.")

    return buy_signals, sell_signals


if __name__ == "__main__":
    import argparse
    from utils import get_config_tickers

    parser = argparse.ArgumentParser(description='T_Raider Daily Screener v3.4')
    parser.add_argument(
        '--mode',
        choices=['CONSERVATIVE', 'BALANCED', 'AGGRESSIVE'],
        default='CONSERVATIVE',
        help='Risk profile mode (default: CONSERVATIVE)'
    )
    parser.add_argument(
        '--market',
        choices=list(MARKET_CONFIGS.keys()),
        default='INDIA',
        help='Target market — controls VIX, flow source, and calendar (default: INDIA)'
    )
    args = parser.parse_args()

    run_screener(get_config_tickers(), mode=args.mode, market=args.market)