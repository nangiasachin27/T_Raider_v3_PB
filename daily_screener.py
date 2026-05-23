"""
daily_screener.py (v3.5 — Critical Bug Fixes)
────────────────────────────────────────────────────────

BUG FIX 1 — get_portfolio_context() used stale entry prices for all holdings
except the current ticker being evaluated. This made total_value wrong, which
made weight_pct wrong, meaning the 20% concentration cap was never accurately
enforced.

FIX: fetch_all_holding_prices() now batch-fetches current market prices for
ALL holdings ONCE before the ticker loop. get_portfolio_context() receives
this pre-built price map and uses real market values for every stock.

BUG FIX 2 — calculate_atr() mixed adjusted Close (prev_close) with
unadjusted High/Low columns. On ex-dividend or split days this created
artificial True Range spikes, overstating ATR and causing undersized
position recommendations.

FIX: calculate_atr() now uses only adjusted columns throughout. If
auto_adjust=True data is used (Close already adjusted), it falls back
cleanly. A new helper _get_price_col() centralises column selection.
"""

import pandas as pd
import json
import os
import datetime
import warnings
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import yfinance as yf

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


# ── Helper: consistent price column selection ─────────────────────────────────

def _get_price_col(df: pd.DataFrame) -> str:
    """
    Returns the best available adjusted close column name.
    Prefers 'Adj Close' (yfinance default), falls back to 'Close'.
    Used consistently by both ATR and signal calculations.
    """
    return 'Adj Close' if 'Adj Close' in df.columns else 'Close'


# ── ATR + position sizing ─────────────────────────────────────────────────────

def calculate_atr(df: pd.DataFrame, window: int = 14) -> float:
    """
    BUG FIX 2: Previously mixed Adj Close (for prev_close) with raw High/Low.
    On ex-dividend or split days this created artificial TR spikes.

    FIX: Use adjusted columns consistently throughout.
    - If 'Adj Close' exists alongside 'High'/'Low', we scale High/Low by the
      adjustment ratio so all three inputs are on the same adjusted basis.
    - If data is already fully adjusted (auto_adjust=True), Close==Adj Close
      and High/Low are already adjusted — the ratio is 1.0, no change.
    """
    if df.empty or len(df) < window + 1:
        return 0.0

    close_col = _get_price_col(df)
    close = df[close_col].copy()

    # Adjust High/Low if raw columns exist alongside Adj Close
    if 'Adj Close' in df.columns and 'Close' in df.columns and 'High' in df.columns:
        # Adjustment ratio: Adj Close / Close — equals 1.0 when already adjusted
        adj_ratio = (df['Adj Close'] / df['Close']).fillna(1.0)
        high = df['High'] * adj_ratio
        low  = df['Low']  * adj_ratio
    elif 'High' in df.columns and 'Low' in df.columns:
        high = df['High']
        low  = df['Low']
    else:
        # Fallback: High/Low not available — use close-to-close as proxy
        high = close
        low  = close

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

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
    avg_vol   = float(volume.iloc[-(avg_window + 1):-1].mean())
    if avg_vol <= 0:
        return True, today_vol, 0.0
    return today_vol >= avg_vol * min_ratio, today_vol, avg_vol


# ── Nifty 50-day high drop ────────────────────────────────────────────────────

def get_nifty_drop_from_50d_high() -> float:
    try:
        nifty = yf.download("^NSEI", period="80d", progress=False, auto_adjust=True)
        if nifty.empty or len(nifty) < 10:
            warnings.warn("Insufficient Nifty data for 50d-high drop — using 0.0.")
            return 0.0
        close    = nifty["Close"].squeeze()
        window   = min(50, len(close))
        high_50d = float(close.iloc[-window:].max())
        latest   = float(close.iloc[-1])
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
        emoji    = "🟢" if self.confidence_tier == "HIGH" else "🟡"
        qty_line  = f"   Suggested Qty: {self.suggested_qty} shares\n" if self.suggested_qty > 0 else ""
        cash_line = f"   Cash Required: ₹{self.cash_required:.0f}\n"   if self.cash_required > 0 else ""
        return (
            f"{emoji} **{self.ticker}** | {self.strategy} | {self.signal_type}\n"
            f"   Price: ₹{self.price:.2f} | Stability: {self.stability_score:.1f}%"
            f" | Expected Return: {self.expected_return:.1f}%\n"
            f"{qty_line}{cash_line}"
            f"   Risk: ₹{self.risk_per_trade_inr:.0f} | Holdings: {self.current_holdings}"
            f" | Weight: {self.portfolio_weight_pct:.1f}%\n"
            f"   {'✅ APPROVED' if self.confidence_tier == 'HIGH' else '⚠️ REVIEW BEFORE ACTING'}"
        )


# ════════════════════════════════════════════════════════════════════════════
# BUG FIX 1 — Portfolio context with accurate market prices
# ════════════════════════════════════════════════════════════════════════════

def fetch_all_holding_prices(holdings: Dict) -> Dict[str, float]:
    """
    BUG FIX 1 (Part A): Batch-fetch current market prices for ALL holdings
    in a single yfinance call BEFORE the ticker loop.

    Previously, get_portfolio_context() used entry_price as a substitute
    for current price for every holding except the one being evaluated.
    This made total_portfolio_value wrong → weight_pct wrong → concentration
    limits not enforced correctly.

    Returns: {ticker: current_price}
    """
    tickers = [t for t, v in holdings.items()
               if (v.get('qty', 0) if isinstance(v, dict) else int(v or 0)) > 0]

    if not tickers:
        return {}

    prices: Dict[str, float] = {}
    try:
        raw = yf.download(tickers, period="2d", progress=False, auto_adjust=True)
        if not raw.empty:
            close = raw["Close"] if "Close" in raw.columns else raw
            if isinstance(close, pd.DataFrame):
                latest = close.iloc[-1]
                for t in tickers:
                    if t in latest.index and pd.notna(latest[t]):
                        prices[t] = float(latest[t])
            else:
                # Single ticker — Series
                prices[tickers[0]] = float(close.iloc[-1])
    except Exception as e:
        warnings.warn(f"Batch holding price fetch failed: {e} — will fall back to entry prices")

    return prices


def get_portfolio_context(
    portfolio: Dict,
    ticker: str,
    latest_price: float,
    holding_prices: Optional[Dict[str, float]] = None,
) -> Tuple[int, float, float, float]:
    """
    BUG FIX 1 (Part B): Use real current market prices for ALL holdings
    when computing total_value and weight_pct.

    Args:
        portfolio       : portfolio dict from load_portfolio()
        ticker          : the stock being evaluated
        latest_price    : current price of this ticker (already fetched)
        holding_prices  : dict of {ticker: price} for all held stocks,
                          pre-fetched by fetch_all_holding_prices().
                          Falls back to entry_price if not provided (old behaviour).

    Returns:
        current_qty      : shares held of this ticker
        weight_pct       : this ticker's weight in portfolio (market value basis)
        total_value      : total portfolio value (cash + all holdings at market price)
        cash             : available cash
    """
    holdings = portfolio.get('holdings', {})
    cash     = float(portfolio.get('cash', 0))

    if holding_prices is None:
        holding_prices = {}

    def _qty(v) -> int:
        return v['qty'] if isinstance(v, dict) else int(v or 0)

    total_market_value = 0.0
    for hticker, hdata in holdings.items():
        qty = _qty(hdata)
        if qty <= 0:
            continue

        if hticker == ticker:
            px = latest_price
        elif hticker in holding_prices:
            # FIX: use real current market price, not stale entry_price
            px = holding_prices[hticker]
        else:
            # Fallback only if batch fetch missed this ticker
            entry = hdata.get('entry_price', 0) if isinstance(hdata, dict) else 0
            px    = entry if entry > 0 else latest_price
            warnings.warn(
                f"No current price for held ticker {hticker} — "
                f"using entry_price ₹{px:.2f} as fallback. "
                f"Concentration limit may be inaccurate."
            )

        total_market_value += qty * px

    total_value  = cash + total_market_value
    raw          = holdings.get(ticker, 0)
    current_qty  = int(raw['qty'] if isinstance(raw, dict) else (raw or 0))
    weight_pct   = (current_qty * latest_price / total_value * 100) if total_value > 0 else 0.0

    return current_qty, weight_pct, total_value, cash


# ── Confidence classification ─────────────────────────────────────────────────

def classify_buy_confidence(stability: float, weight_pct: float, current_qty: int,
                             cash: float, estimated_cost: float,
                             max_weight: float = 20.0) -> Tuple[str, str]:
    if estimated_cost > cash * 1.05:
        return "SKIP", f"Insufficient cash (Need ₹{estimated_cost:.0f}, Have ₹{cash:.0f})"
    if current_qty > 0:
        if stability >= 70 and weight_pct < max_weight / 2:
            return "MEDIUM", f"Already holding {current_qty} shares — small add possible"
        return "SKIP", f"Already holding {current_qty} shares — avoid concentration"
    if stability >= 70 and weight_pct < max_weight:
        return "HIGH", "Strong MC validation + within concentration limits"
    if stability >= 60 and weight_pct < max_weight:
        return "MEDIUM", "Marginal stability score — review recommended"
    if weight_pct >= max_weight:
        return "SKIP", f"Concentration limit ({weight_pct:.1f}% >= {max_weight}%)"
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
        "date":      date_str,
        "timestamp": datetime.datetime.now().isoformat(),
        "market_regime": regime_info,
        "portfolio_snapshot": {
            "cash":            portfolio.get('cash', 0),
            "holdings_count":  len(portfolio.get('holdings', {})),
        },
        "total_signals":    len(signals),
        "high_confidence":  len([s for s in signals if s.confidence_tier == "HIGH"]),
        "medium_confidence": len([s for s in signals if s.confidence_tier == "MEDIUM"]),
        "recommendations":  [s.to_dict() for s in signals if s.confidence_tier != "SKIP"],
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

    macro_snap = regime_info.get("macro", {})
    if macro_snap:
        vix_str = (f"VIX {macro_snap['vix']:.1f} [{macro_snap['vix_status']}]"
                   if macro_snap.get('vix') else "VIX N/A")
        flow_str = (f"Flow {macro_snap['inst_flow']:+,.0f}M [{macro_snap['flow_status']}]"
                    if macro_snap.get('inst_flow') is not None else "Flow N/A")
        ov_str   = (f"Overnight {macro_snap['overnight_pct']:+.1f}%"
                    if macro_snap.get('overnight_pct') is not None else "Overnight N/A")
        print(f"**Macro:** {vix_str} | {flow_str} | {ov_str}")
        if macro_snap.get('high_risk_day'):
            print(f"⚠️ HIGH-RISK DAY: {macro_snap['high_risk_reason']}")

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

    print(f"\n--- T_Raider Hybrid Execution Engine (v3.5 — Bug Fix Release) ---")
    print(f"Timestamp : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode      : {mode}")
    print(f"Market    : {market}")
    if volume_min_ratio > 0:
        print(f"  Volume filter: BUY signals require ≥{volume_min_ratio*100:.0f}% of 20-day avg volume")

    RISK_PROFILES = {
        "CONSERVATIVE": {"allow_mean_reversion": False, "nifty_drop_threshold": 0.0},
        "BALANCED":     {"allow_mean_reversion": True,  "nifty_drop_threshold": 0.05},
        "AGGRESSIVE":   {"allow_mean_reversion": True,  "nifty_drop_threshold": 0.03},
    }
    config = RISK_PROFILES.get(mode, RISK_PROFILES["CONSERVATIVE"])
    print(f"  Mean-reversion in downtrend: {config['allow_mean_reversion']}"
          f" (threshold: {config['nifty_drop_threshold']:.0%})")

    # ── Gate 1: Market Regime ─────────────────────────────────────────────
    print("\n🌐 Checking market regime…")
    is_uptrend, nifty_close, nifty_ema = get_market_regime()
    nifty_drop = 0.0
    if not is_uptrend and config["allow_mean_reversion"]:
        nifty_drop = get_nifty_drop_from_50d_high()
        print(f"  📐 Nifty 50-day high drop: {nifty_drop:.1%} "
              f"(threshold: {config['nifty_drop_threshold']:.0%})")

    regime_info = {
        "is_uptrend":          is_uptrend,
        "nifty_close":         round(nifty_close, 2),
        "nifty_ema50":         round(nifty_ema, 2),
        "nifty_drop_50d_high": round(nifty_drop, 4),
        "summary":             regime_summary(is_uptrend, nifty_close, nifty_ema),
    }
    print(f"  {regime_info['summary']}")
    if not is_uptrend:
        if config["allow_mean_reversion"]:
            print(f"  ⛔ Downtrend — mean-reversion BUYs allowed only if "
                  f"Nifty drop > {config['nifty_drop_threshold']:.0%} "
                  f"(current: {nifty_drop:.1%})")
        else:
            print("  ⛔ Downtrend detected — BUY signals suppressed.")

    # ── Gate 1b: Macro Environment Filter ────────────────────────────────
    print("\n🌍 Fetching macro environment data…")
    if market not in MARKET_CONFIGS:
        print(f"  ⚠️ Unknown market '{market}' — macro filter disabled.")
        macro = macro_context = None
    else:
        macro         = MacroFilter(MARKET_CONFIGS[market])
        macro_context = macro.run()
        macro.print_summary(macro_context)
        regime_info["macro"] = {
            "vix":             macro_context.vix,
            "vix_status":      macro_context.vix_status,
            "inst_flow":       macro_context.institutional_flow,
            "flow_status":     macro_context.flow_status,
            "overnight_pct":   macro_context.overnight_change_pct,
            "high_risk_day":   macro_context.is_high_risk_day,
            "high_risk_reason": macro_context.high_risk_reason,
        }

    # ── Sector Momentum ───────────────────────────────────────────────────
    print("\n📊 Loading Sector Momentum Data…")
    try:
        with open('config/dynamic_sector_map.json', 'r') as f:
            dynamic_map = json.load(f)
        sector_ranks = get_sector_ranks()
        print("  ✅ Sector ranks loaded.")
    except FileNotFoundError:
        print("  ⚠️ Sector map missing. Run sector_mapper.py first. Sector boosting disabled.")
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

    all_signals: List[ScreenerSignal] = []
    buy_signals  = []
    sell_signals = []

    portfolio = load_portfolio()

    # ════════════════════════════════════════════════════════════════════════
    # BUG FIX 1: Batch-fetch current prices for ALL holdings ONCE,
    # before the per-ticker loop. This gives get_portfolio_context()
    # real market values instead of stale entry prices.
    # ════════════════════════════════════════════════════════════════════════
    holdings = portfolio.get('holdings', {})
    print(f"\n💰 Pre-fetching current prices for {len(holdings)} held positions…")
    holding_prices = fetch_all_holding_prices(holdings)
    print(f"  ✅ Got live prices for {len(holding_prices)}/{len(holdings)} holdings.")

    vol_filtered    = 0
    macro_skipped   = 0
    macro_downgraded = 0

    for ticker in tickers:
        plan = optimized_params.get(ticker)
        if not plan or plan.get('strategy') == "NONE":
            continue

        df = get_stock_data(full_market_data, ticker)
        if df.empty or len(df) < 200:
            continue

        price_col    = _get_price_col(df)   # BUG FIX 2: consistent column selection
        price        = df[price_col]
        latest_price = float(price.iloc[-1])

        strat_type = plan['strategy']
        p          = plan.get('params', {})

        if   strat_type == "TREND":     res = apply_golden_cross_strategy(price)
        elif strat_type == "RSI":       res = apply_rsi_strategy(price, window=p.get('window', 14), buy=p.get('buy', 30), sell=p.get('sell', 70))
        elif strat_type == "VOLATILITY": res = apply_bollinger_strategy(price)
        elif strat_type == "BREAKOUT":  res = apply_breakout_strategy(price, window=p.get('window', 20))
        elif strat_type == "MACD":      res = apply_macd_strategy(price)
        elif strat_type == "STRETCH":   res = apply_stretch_strategy(price, window=p.get('window', 20), threshold=p.get('threshold', 0.05))
        else:
            continue

        latest_signal = res.iloc[-1]['Signal']
        if latest_signal not in (1, -1):
            continue

        # BUG FIX 1: Pass holding_prices so all positions use real market values
        current_qty, weight_pct, portfolio_value, cash = get_portfolio_context(
            portfolio, ticker, latest_price, holding_prices=holding_prices
        )

        # Capital override
        override_path = Path("config/capital_override.json")
        if override_path.exists() and capital is None:
            with open(override_path) as f:
                effective_capital = json.load(f).get("total_baseline_wealth", portfolio_value)
        else:
            effective_capital = capital if capital is not None else portfolio_value

        stability = float(plan.get('stability_score', 0) or 0)

        # ── BUY signal pipeline ───────────────────────────────────────────
        if latest_signal == 1:

            # Gate 1: Regime
            if not is_uptrend:
                if config["allow_mean_reversion"]:
                    if nifty_drop > config["nifty_drop_threshold"] and strat_type in ['RSI', 'VOLATILITY', 'STRETCH']:
                        pass  # allowed — fall through to Gate 1b
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

            # Gate 1b: Macro
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
                    macro_downgraded    += 1
                    macro_downgrade_reason = " | ".join(macro_eval.warning_flags())

            # Gate 2: Volume
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
                                f"Today: {today_vol:,.0f} | Avg20: {avg_vol:,.0f}"),
                    ))
                    continue
            else:
                today_vol, avg_vol = 0.0, 0.0

            # Gate 3: ATR sizing + confidence
            # BUG FIX 2: calculate_atr() now uses adjusted columns consistently
            atr            = calculate_atr(df)
            suggested_qty  = calculate_position_size(effective_capital, atr)
            estimated_cost = suggested_qty * latest_price

            confidence_tier, reason = classify_buy_confidence(
                stability, weight_pct, current_qty, cash, estimated_cost
            )

            if macro_downgrade_reason and confidence_tier == "HIGH":
                confidence_tier = "MEDIUM"
                reason = f"{reason} | {macro_downgrade_reason}"
            elif macro_downgrade_reason:
                reason = f"{reason} | {macro_downgrade_reason}"

            # Sector momentum
            stock_sector_info = dynamic_map.get(ticker, {"nse_index": "UNKNOWN", "yf_sector": "Unknown"})
            nse_index      = stock_sector_info["nse_index"]
            yf_sector_name = stock_sector_info["yf_sector"]
            sector_data    = sector_ranks.get(nse_index, {"rank": 99, "is_outperforming": False, "rs_score": 0.0})
            is_sector_strong = sector_data['is_outperforming']
            sector_rs        = sector_data['rs_score']
            base_score       = plan.get('composite_score', 0) or plan.get('expected_return', 0)
            final_score      = base_score * (1.2 if is_sector_strong else 0.8)
            sector_icon      = "🔥" if is_sector_strong else "❄️"
            vol_ratio        = (today_vol / avg_vol * 100) if avg_vol > 0 else 0
            vol_note         = f"Vol: {vol_ratio:.0f}% avg" if avg_vol > 0 else ""
            aug_reason       = (f"{reason} | {yf_sector_name[:10]} {sector_icon}"
                                + (f" | {vol_note}" if vol_note else ""))

            all_signals.append(ScreenerSignal(
                ticker=ticker, signal_type="BUY", price=latest_price,
                strategy=strat_type, expected_return=plan.get('expected_return', 0),
                stability_score=stability, confidence_tier=confidence_tier,
                suggested_qty=suggested_qty, risk_per_trade_inr=effective_capital * 0.01,
                current_holdings=current_qty, portfolio_weight_pct=weight_pct,
                reason=aug_reason, atr_14d=atr, cash_required=estimated_cost,
            ))
            buy_signals.append({
                "ticker":          ticker,
                "price":           latest_price,
                "reason":          strat_type,
                "sector":          yf_sector_name[:12],
                "sector_rs":       sector_rs,
                "is_sector_strong": is_sector_strong,
                "expected_return": plan.get('expected_return', 0),
                "stability":       stability,
                "sharpe_ratio":    plan.get('sharpe_ratio', None),
                "max_drawdown":    plan.get('max_drawdown', None),
                "composite_score": final_score,
                "folds_passed":    plan.get('folds_passed', None),
                "volume_ratio":    round(vol_ratio, 1) if avg_vol > 0 else None,
            })

        # ── SELL signal pipeline ──────────────────────────────────────────
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
    if vol_filtered:
        print(f"\n🔇 Volume filter blocked {vol_filtered} BUY signal(s) on thin volume.")
    if macro_skipped:
        print(f"🌍 Macro filter BLOCKED {macro_skipped} BUY signal(s).")
    if macro_downgraded:
        print(f"🌍 Macro filter DOWNGRADED {macro_downgraded} BUY signal(s) HIGH → MEDIUM.")

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
            rs_str = f"{b['sector_rs']:>6.1f}%" if b['sector_rs'] != 0.0 else "N/A"
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

    parser = argparse.ArgumentParser(description='T_Raider Daily Screener v3.5')
    parser.add_argument('--mode',   choices=['CONSERVATIVE', 'BALANCED', 'AGGRESSIVE'], default='CONSERVATIVE')
    parser.add_argument('--market', choices=list(MARKET_CONFIGS.keys()), default='INDIA')
    args = parser.parse_args()

    run_screener(get_config_tickers(), mode=args.mode, market=args.market)