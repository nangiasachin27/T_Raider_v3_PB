"""
daily_screener_macro_patch.py
═══════════════════════════════════════════════════════════════════════════════
This file is NOT a standalone module. It shows EXACTLY what to add/change
in daily_screener.py to wire in macro_filter.py.

Three sections:
    A) New imports to add at the top
    B) Session-level fetch to add inside run_screener(), before the ticker loop
    C) Per-ticker gate to add inside the BUY block, after Gate 1 (regime)

Search for each ── PATCH POINT ── comment in daily_screener.py and apply
the corresponding block below.
═══════════════════════════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────────────────────
# A) ADD THESE IMPORTS at the top of daily_screener.py
#    (after the existing imports block)
# ─────────────────────────────────────────────────────────────────────────────

from macro_filter import (
    MacroFilter,
    MARKET_CONFIGS,
    apply_macro_filter_to_signal,
    FilterAction,
)


# ─────────────────────────────────────────────────────────────────────────────
# B) ADD THIS BLOCK inside run_screener(), after the sector momentum block
#    and before the "Load optimal params" block.
#
#    run_screener() signature change — add `market` parameter:
#
#    def run_screener(tickers, capital=None, min_stability=60.0,
#                     volume_min_ratio=0.80, mode="CONSERVATIVE",
#                     market="INDIA"):          ← ADD THIS
# ─────────────────────────────────────────────────────────────────────────────

def _patch_B_macro_fetch_block(market: str):
    """
    ── PATCH POINT B ── inside run_screener(), after sector momentum block
    ─────────────────────────────────────────────────────────────────────
    Paste the code INSIDE the triple-quotes into daily_screener.run_screener()
    at the indicated location. Remove the outer function wrapper.
    """
    CODE = """
    # ── Macro Environment Filter ──────────────────────────────────────────
    # Fetched once per session — not once per ticker.
    # All three filters (VIX, institutional flow, earnings) run from this
    # single context object. Failures degrade gracefully to PASS.
    print("\\n🌍 Fetching macro environment data…")
    if market not in MARKET_CONFIGS:
        print(f"  ⚠️  Unknown market '{market}' — macro filter disabled.")
        macro         = None
        macro_context = None
    else:
        macro         = MacroFilter(MARKET_CONFIGS[market])
        macro_context = macro.run()
        macro.print_summary(macro_context)
    """
    return CODE


# ─────────────────────────────────────────────────────────────────────────────
# C) ADD THIS BLOCK inside the BUY gate, AFTER Gate 1 (regime check)
#    and BEFORE Gate 2 (volume confirmation).
#
#    Search for the comment:
#        # Gate 2: Volume confirmation
#    and insert the block immediately before it.
# ─────────────────────────────────────────────────────────────────────────────

def _patch_C_per_ticker_gate():
    """
    ── PATCH POINT C ── inside the BUY block, before Gate 2: Volume
    ────────────────────────────────────────────────────────────────
    Paste the code INSIDE the triple-quotes. Remove the outer function wrapper.
    """
    CODE = """
            # Gate 1b: Macro environment filter
            # Runs after the regime gate (Gate 1) and before volume (Gate 2).
            # Checks VIX, institutional flow, earnings proximity, calendar,
            # and overnight reference market in one call.
            if macro is not None and macro_context is not None:
                from macro_filter import FilterAction
                macro_eval = macro.evaluate_buy(ticker, macro_context)

                if macro_eval.action == FilterAction.SKIP:
                    all_signals.append(ScreenerSignal(
                        ticker=ticker, signal_type="BUY", price=latest_price,
                        strategy=strat_type,
                        expected_return=plan.get('expected_return', 0),
                        stability_score=stability,
                        confidence_tier="SKIP",
                        suggested_qty=0, risk_per_trade_inr=0,
                        current_holdings=current_qty,
                        portfolio_weight_pct=weight_pct,
                        reason=macro_eval.reason,
                    ))
                    continue

                # DOWNGRADE: signal continues but confidence is capped at MEDIUM
                # later when classify_buy_confidence() is called. Store the flag.
                macro_downgrade_reason = (
                    " | ".join(macro_eval.warning_flags())
                    if macro_eval.action == FilterAction.DOWNGRADE
                    else ""
                )
            else:
                macro_downgrade_reason = ""
    """
    return CODE


# ─────────────────────────────────────────────────────────────────────────────
# D) APPLY DOWNGRADE after classify_buy_confidence()
#
#    Find the block that calls classify_buy_confidence():
#        confidence_tier, reason = classify_buy_confidence(...)
#    And add these lines immediately after it.
# ─────────────────────────────────────────────────────────────────────────────

def _patch_D_apply_downgrade():
    """
    ── PATCH POINT D ── immediately after classify_buy_confidence() call
    ────────────────────────────────────────────────────────────────────
    """
    CODE = """
            # Apply macro downgrade if flagged in Gate 1b
            if macro_downgrade_reason and confidence_tier == "HIGH":
                confidence_tier = "MEDIUM"
                reason = f"{reason} | {macro_downgrade_reason}"
    """
    return CODE


# ─────────────────────────────────────────────────────────────────────────────
# E) UPDATE __main__ block at the bottom of daily_screener.py
#    Add --market argument to argparse
# ─────────────────────────────────────────────────────────────────────────────

MAIN_BLOCK_REPLACEMENT = """
if __name__ == "__main__":
    import argparse
    from utils import get_config_tickers

    parser = argparse.ArgumentParser(description='T_Raider Daily Screener')
    parser.add_argument(
        '--mode',
        choices=['CONSERVATIVE', 'BALANCED', 'AGGRESSIVE'],
        default='CONSERVATIVE',
        help='Risk profile mode'
    )
    parser.add_argument(
        '--market',
        choices=['INDIA', 'AUSTRALIA', 'CANADA', 'USA'],
        default='INDIA',
        help='Target market (determines VIX, flow source, calendar events)'
    )
    args = parser.parse_args()

    run_screener(get_config_tickers(), mode=args.mode, market=args.market)
"""


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE — full list of changes to daily_screener.py
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_SUMMARY = """
CHANGES TO daily_screener.py
─────────────────────────────────────────────────────────────────────────────
1. Imports (top of file)
   ADD: from macro_filter import MacroFilter, MARKET_CONFIGS,
                                apply_macro_filter_to_signal, FilterAction

2. run_screener() signature
   ADD `market: str = "INDIA"` parameter

3. After sector momentum block (before "Load optimal params")
   ADD: macro + macro_context fetch block  [Patch B]

4. Inside BUY gate, after Gate 1 regime check, before Gate 2 volume check
   ADD: Gate 1b macro evaluation block     [Patch C]

5. After classify_buy_confidence() call
   ADD: Apply macro downgrade              [Patch D]

6. __main__ block
   ADD: --market argparse argument         [Patch E]
─────────────────────────────────────────────────────────────────────────────
To run for Australia:
    python daily_screener.py --mode CONSERVATIVE --market AUSTRALIA

To run for India (unchanged default):
    python daily_screener.py --mode CONSERVATIVE
─────────────────────────────────────────────────────────────────────────────
"""

if __name__ == "__main__":
    print(CHANGE_SUMMARY)