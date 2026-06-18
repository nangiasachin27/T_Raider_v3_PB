"""
autopilot/quarterly_manager.py
──────────────────────────────
Quarterly profit-booking + compounding engine.
Hybrid: Profit target + Trailing stop + Min/Max time caps.

BUG FIX 6 — QuarterState.save() silently dropped any key in
quarterly_config.json that was not explicitly loaded in __init__().

Root cause: save() serialises self.__dict__, which only contains fields
assigned as self.X = ... in __init__(). Any top-level key in the JSON
not listed there (e.g. "chaser") is loaded from file but never stored on
the object, so it disappears on the next save().

FIX: Two changes:
  1. __init__ now accepts and stores ALL unknown keys via self._extra.
  2. save() merges self._extra back into the dict before writing,
     so no key is ever silently dropped regardless of future additions.
  3. Added self.chaser = kwargs.get('chaser', {}) explicitly as the most
     common known extra key, with a clear comment to add future keys here.
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from market_hours import guard_or_exit

CONFIG_PATH           = Path("config/quarterly_config.json")
BROKER_CONFIG_PATH    = Path("config/broker_config.json")
CAPITAL_OVERRIDE_PATH = Path("config/capital_override.json")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autopilot.bot import run_autopilot_cycle
from execution.adapters.base import ExecutionAdapter
from execution.adapters.paper_adapter import PaperExecutionAdapter


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA MODELS
# ═════════════════════════════════════════════════════════════════════════════

# Known top-level keys in quarterly_config.json.
# Add new keys here as explicit self.X = kwargs.get('X', default) lines
# so they are readable by name elsewhere in the codebase.
_KNOWN_KEYS = {
    'enabled', 'profit_target_pct','super_target_pct', 'quarter_days', 'min_quarter_days',
    'trailing_stop_pct', 'compound_mode', 'original_capital',
    'current_base_capital', 'highest_value', 'quarter_start_date',
    'quarter_number', 'realized_pnl_history', 'last_harvest_date',
    'paper_trading', 'broker', 'chaser','market'
}


class QuarterState:
    def __init__(self, **kwargs):
        # ── Explicit known fields ─────────────────────────────────────────
        self.enabled              = kwargs.get('enabled', True)
        self.profit_target_pct    = kwargs.get('profit_target_pct', 0.05)
        self.super_target_pct     = kwargs.get('super_target_pct', None)
        self.quarter_days         = kwargs.get('quarter_days', 90)
        self.min_quarter_days     = kwargs.get('min_quarter_days', 30)
        self.trailing_stop_pct    = kwargs.get('trailing_stop_pct', 0.10)
        self.compound_mode        = kwargs.get('compound_mode', True)
        self.original_capital     = kwargs.get('original_capital', 100000.0)
        self.current_base_capital = kwargs.get('current_base_capital', 100000.0)
        self.highest_value        = kwargs.get('highest_value', self.current_base_capital)
        self.quarter_start_date   = kwargs.get('quarter_start_date', datetime.now().isoformat())
        self.quarter_number       = kwargs.get('quarter_number', 1)
        self.realized_pnl_history = kwargs.get('realized_pnl_history', [])
        self.last_harvest_date    = kwargs.get('last_harvest_date', None)
        self.paper_trading        = kwargs.get('paper_trading', True)
        self.broker               = kwargs.get('broker', 'paper')
        self.market               = kwargs.get('market', 'INDIA')

        # FIX 6: Explicitly load chaser block — most common "extra" key.
        # Add any future top-level config keys here in the same pattern.
        self.chaser = kwargs.get('chaser', {})

        # FIX 6: Catch-all for any OTHER unknown keys present in the JSON.
        # Stored in _extra and merged back into the file on save(),
        # so nothing is ever silently dropped.
        self._extra = {
            k: v for k, v in kwargs.items()
            if k not in _KNOWN_KEYS
        }
        if self._extra:
            unknown = list(self._extra.keys())
            print(
                f"  ℹ️  QuarterState: {len(unknown)} unknown config key(s) preserved: "
                f"{unknown}. Add explicit self.X = kwargs.get('{unknown[0]}', ...) "
                f"lines in __init__ if you need to access them by name."
            )

    @classmethod
    def from_file(cls, path: Path = CONFIG_PATH):
        if not path.exists():
            raise FileNotFoundError(f"❌ {path} not found. Create it first.")
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path: Path = CONFIG_PATH):
        """
        FIX 6: Serialise ALL fields — both explicitly named ones and any
        extra keys captured in self._extra — so nothing is ever dropped.

        Build order:
          1. Start with self._extra (unknown keys from original file)
          2. Overlay self.__dict__ (all named fields)
          3. Remove the internal _extra key itself before writing
        """
        data = {}
        # Step 1: preserve unknown keys
        data.update(self._extra)
        # Step 2: overlay all named instance fields
        data.update(self.__dict__)
        # Step 3: remove internal bookkeeping key
        data.pop('_extra', None)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def inject_capital(self, path: Path = CAPITAL_OVERRIDE_PATH):
        with open(path, "w") as f:
            json.dump({"total_baseline_wealth": self.current_base_capital}, f)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: FACTORY
# ═════════════════════════════════════════════════════════════════════════════

def get_execution_adapter(state: QuarterState) -> ExecutionAdapter:
    if state.paper_trading or state.broker == 'paper':
        print("📋 MODE: Paper Trading (no real orders)")
        return PaperExecutionAdapter()

    if state.broker == 'upstox':
        if not BROKER_CONFIG_PATH.exists():
            raise FileNotFoundError(f"❌ {BROKER_CONFIG_PATH} missing.")
        print("🏦 MODE: Upstox Broker")
        from execution.adapters import get_upstox_adapter
        UpstoxAdapter = get_upstox_adapter()
        return UpstoxAdapter(BROKER_CONFIG_PATH)

    raise ValueError(f"Unknown broker: {state.broker}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: HARVEST ENGINE (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

class HarvestEngine:
    def __init__(self, state: QuarterState, adapter: ExecutionAdapter):
        self.state   = state
        self.adapter = adapter

    def check_trigger(self) -> Tuple[bool, str]:
        if not self.state.enabled:
            return False, "Quarterly system disabled."

        start   = datetime.fromisoformat(self.state.quarter_start_date)
        elapsed = (datetime.now() - start).days

        snapshot      = self.adapter.get_portfolio_snapshot()
        current_value = snapshot.total_value
        base          = self.state.current_base_capital
        target        = self.state.profit_target_pct
        super_target  = self.state.super_target_pct  
        min_days      = self.state.min_quarter_days
        max_days      = self.state.quarter_days
        trail_pct     = self.state.trailing_stop_pct

        peak = max(self.state.highest_value, current_value)
        if peak > self.state.highest_value:
            self.state.highest_value = peak

        current_return       = (current_value - base) / base if base > 0 else 0
        trailing_stop_level = max(
                    peak * (1 - trail_pct),   # normal trailing stop
                    base,                      # floor: never stop out below starting capital
        )
        drawdown_from_peak   = (peak - current_value) / peak if peak > 0 else 0
        floor_active = (base > peak * (1 - trail_pct))
        stop_label   = f"₹{trailing_stop_level:,.2f} ({trail_pct*100:.0f}% below peak)" \
                    + (" ← FLOOR ACTIVE (base capital protected)" if floor_active else "")

        print(f"\n{'='*60}")
        print(f"📊 QUARTER {self.state.quarter_number} STATUS")
        print(f"{'='*60}")
        print(f" Start Date    : {start.date()}")
        print(f" Elapsed Days  : {elapsed} / {max_days} (min: {min_days})")
        print(f" Base Capital  : ₹{base:,.2f}")
        print(f" Cash          : ₹{snapshot.cash:,.2f}")
        print(f" Market Value  : ₹{snapshot.market_value:,.2f}")
        print(f" Total Value   : ₹{current_value:,.2f}")
        print(f" Quarter High  : ₹{peak:,.2f}")
        print(f" Trailing Stop : {stop_label}")
        print(f" Current Return: {current_return*100:+.2f}%")
        print(f" Target Return : +{target*100:.0f}%")
        print(f" Compound Mode : {'ON' if self.state.compound_mode else 'OFF'}")
        print(f" Broker        : {self.state.broker} ({'paper' if self.state.paper_trading else 'live'})")

        # ── Rule 0 (NEW): Super Target ────────────────────────────────────────
        # If super_target_pct is configured AND current return has met or
        # exceeded it, harvest IMMEDIATELY — no minimum hold period gate.
        #
        # Use case: an exceptional short-term run (+10% in 7 days) that is
        # worth booking now rather than risking a reversal over 30 days.
        #
        # super_target_pct must be GREATER than profit_target_pct.
        # If not set (None), this rule is simply skipped.
        if super_target is not None and current_return >= super_target:
            return True, (
                f"🚀 SUPER TARGET hit! {current_return*100:+.2f}% >= "
                f"+{super_target*100:.0f}% — harvesting immediately "
                f"(bypassing {min_days}d minimum hold, only {elapsed}d elapsed)"
            )
        if elapsed < min_days:
            self.state.save()
            return False, f"⏳ Minimum hold: {elapsed}/{min_days} days. {min_days - elapsed} days until harvest allowed."

        if current_return >= target:
            return True, f"🎯 Profit target hit! {current_return*100:+.2f}% >= +{target*100:.0f}%"

        if current_value <= trailing_stop_level and current_value < peak:
            return True, (
                f"🛑 Trailing stop hit! "
                f"Down {drawdown_from_peak*100:.1f}% from peak ₹{peak:,.0f} "
                f"(stop was ₹{trailing_stop_level:,.0f})"
            )

        if elapsed >= max_days:
            return True, f"⏰ Max time cap reached ({elapsed} days >= {max_days})"

        self.state.save()
        days_left = max_days - elapsed
        cushion   = (current_value - trailing_stop_level) / base * 100 if base > 0 else 0
        return False, (
            f"⏳ Hold. {days_left} days left. "
            f"Need +{(target - current_return)*100:.2f}% for target. "
            f"Trailing cushion: {cushion:+.2f}%"
        )

    def execute(self) -> QuarterState:
        print(f"\n{'='*60}")
        print(f"🌾 QUARTER {self.state.quarter_number} HARVEST")
        print(f"{'='*60}")

        snapshot = self.adapter.get_portfolio_snapshot()
        base     = self.state.current_base_capital

        self._liquidate_all(snapshot.holdings)

        post_snapshot = self.adapter.get_portfolio_snapshot()
        post_value    = post_snapshot.total_value
        realized_pnl  = post_value - base
        realized_pct  = (realized_pnl / base * 100) if base > 0 else 0

        if self.state.compound_mode:
            new_base = post_value
            print(f"\n🔄 COMPOUND: Reinvesting ₹{new_base:,.2f}")
        else:
            new_base = self.state.original_capital
            booked   = post_value - self.state.original_capital
            print(f"\n📤 RESET: Returning to original ₹{new_base:,.2f}")
            if booked > 0:
                print(f" 💵 Profit to withdraw: ₹{booked:,.2f}")
            elif booked < 0:
                print(f" 🔴 Loss absorbed: ₹{abs(booked):,.2f}")

        self.state.quarter_number       += 1
        self.state.current_base_capital  = round(new_base, 2)
        self.state.highest_value         = round(new_base, 2)
        self.state.quarter_start_date    = datetime.now().isoformat()
        self.state.last_harvest_date     = datetime.now().isoformat()

        self.state.realized_pnl_history.append({
            "quarter":       self.state.quarter_number - 1,
            "start_capital": base,
            "end_value":     round(post_value, 2),
            "realized_pnl":  round(realized_pnl, 2),
            "return_pct":    round(realized_pct, 2),
            "harvest_date":  self.state.last_harvest_date,
            "compound_mode": self.state.compound_mode,
        })

        print(f"\n{'='*60}")
        print(f"✅ HARVEST COMPLETE — Q{self.state.quarter_number - 1}")
        print(f" Realized P&L : ₹{realized_pnl:,.2f} ({realized_pct:+.2f}%)")
        print(f" Next Base    : ₹{new_base:,.2f}")
        print(f" Next Quarter : #{self.state.quarter_number}")
        print(f" Start Date   : {self.state.quarter_start_date[:10]}")
        print(f"{'='*60}")

        return self.state

    def _liquidate_all(self, holdings: Dict) -> None:
        if not holdings:
            print("📭 No positions to liquidate.")
            return

        print(f"\n🔴 LIQUIDATING {len(holdings)} POSITION(S)")

        for ticker, data in list(holdings.items()):
            qty = data.get('qty', 0) if isinstance(data, dict) else int(data or 0)
            if qty <= 0:
                continue

            ltp = self.adapter.get_ltp(ticker)
            print(f" 🟢 {ticker}: {qty} shares @ ₹{ltp:.2f}")

            result = self.adapter.place_market_order(
                ticker=ticker, qty=qty, side="SELL",
                tag="QuarterlyHarvest", price=ltp,
            )
            if result.success:
                print(f" ✅ Order {result.order_id} | Status: {result.status}")
            else:
                print(f" ❌ FAILED: {result.message}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: ACTIVE PROFIT ENGINE INTEGRATION (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def _run_active_engine(state: QuarterState) -> Dict:
    from autopilot.active_manager import ActiveProfitEngine

    print(f"\n{'─'*60}")
    print("🎯 ACTIVE PROFIT ENGINE")
    print(f"{'─'*60}")

    engine = ActiveProfitEngine()
    result = engine.run()

    with open("config/.active_risk_mult.json", "w") as f:
        json.dump({"risk_multiplier": result["risk_multiplier"]}, f)

    return result


def _print_active_results(result: Dict):
    m = result["milestone"]
    print(f"\n📊 Quarter Status")
    print(f"   Base Capital: ₹{result['base_capital']:,.0f}")
    print(f"   Target: +{result['target_pct']:.1f}%")
    print(f"   Current Return: {result['current_return_pct']:+.2f}%")
    print(f"   Days: {result['days_elapsed']}/{result['quarter_days']}")
    print(f"   Milestone: {m['action']} — {m['message']}")
    print(f"   Risk Multiplier: {result['risk_multiplier']:.1f}x")

    if result.get("per_stock_targets"):
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

    if result.get("rebalances"):
        print(f"\n🔄 Rebalances ({len(result['rebalances'])}):")
        for r in result["rebalances"]:
            print(f"   {r['ticker']} → {r['replacement']} ({r['score']:.1f} vs {r['replacement_score']:.1f})")

    if result["push_analysis"]:
        print(f"\n⚡ Push Analysis:")
        print(f"   {result['push_analysis']['note']}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

class QuarterlyManager:
    def __init__(self, mode: str = "CONSERVATIVE", filters: str = "ALL"):
        self.mode    = mode
        self.filters = filters
        self.state   = QuarterState.from_file()
        self.adapter = get_execution_adapter(self.state)

    def run(self) -> None:
        print("\n" + "=" * 70)
        print("🚀 T_RAIDER QUARTERLY COMPOUNDING ENGINE")
        print(f" Mode: {self.mode}")
        print("=" * 70)
        # ── Market hours gate ─────────────────────────────────────────────
        # Reads market from quarterly_config.json broker field.
        # sys.exit(0) if closed so GitHub Actions does not report failure.
        market = getattr(self.state, 'market', 'INDIA')
        from market_hours import guard_or_exit
        guard_or_exit(market, exit_on_closed=True)

        engine               = HarvestEngine(self.state, self.adapter)
        should_harv, message = engine.check_trigger()
        print(f"\n{message}")

        if should_harv:
            self.state = engine.execute()
            self.state.save()          # FIX 6: chaser block preserved
            self.state.inject_capital()
            print("\n⏸️  Harvest complete. Run again to start new quarter.")
            return

        active_result = _run_active_engine(self.state)
        _print_active_results(active_result)

        self.state.inject_capital()
        print(f"\n🤖 Running daily autopilot cycle...")
        run_autopilot_cycle(mode=self.mode,market=market, filters=self.filters)

        post_snapshot = self.adapter.get_portfolio_snapshot()
        if post_snapshot.total_value > self.state.highest_value:
            self.state.highest_value = round(post_snapshot.total_value, 2)
            self.state.save()          # FIX 6: chaser block preserved
            print(f"\n📈 New quarter high: ₹{self.state.highest_value:,.2f}")

        print("\n✅ Daily cycle complete.")


def main():
    parser = argparse.ArgumentParser(description="T_Raider Quarterly Manager")
    parser.add_argument("--mode", default="AUTO", 
                       choices=["AUTO", "CONSERVATIVE", "BALANCED", "AGGRESSIVE"],
                       help="Trading mode: AUTO = auto-select based on performance")
    # ADD THE FILTERS ARGUMENT
    parser.add_argument("--filters", default="ALL",
                       help="Comma-separated filters to enable: ALL, NONE, REGIME, MACRO, VOLUME, CORRELATION")
    args = parser.parse_args()

    # AUTO mode selection
    if args.mode == "AUTO":
        from autopilot.auto_mode import auto_select_mode
        selected_mode, reason = auto_select_mode()
        print(f"\n{'='*60}")
        print("🎛️ AUTO MODE SELECTION")
        print(f"{'='*60}")
        print(f"Selected: {selected_mode}")
        print(f"Reason: {reason}")
        print(f"{'='*60}\n")
    else:
        selected_mode = args.mode

    manager = QuarterlyManager(mode=selected_mode, filters=args.filters)
    manager.run()


if __name__ == "__main__":
    main()