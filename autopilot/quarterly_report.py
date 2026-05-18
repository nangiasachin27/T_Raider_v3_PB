"""Usage: python autopilot/quarterly_report.py"""
import json
from pathlib import Path

CONFIG = Path("config/quarterly_config.json")

def main():
    if not CONFIG.exists():
        print("❌ No quarterly config.")
        return
    
    with open(CONFIG) as f:
        cfg = json.load(f)
    
    history = cfg.get('realized_pnl_history', [])
    
    print("\n" + "=" * 70)
    print("📈 T_RAIDER QUARTERLY REPORT")
    print(f"{'='*70}")
    print(f"Current Q      : #{cfg['quarter_number']}")
    print(f"Compound       : {'ON' if cfg['compound_mode'] else 'OFF'}")
    print(f"Original Cap   : ₹{cfg['original_capital']:,.2f}")
    print(f"Current Base   : ₹{cfg['current_base_capital']:,.2f}")
    print(f"Broker         : {cfg.get('broker', 'paper')} ({'paper' if cfg.get('paper_trading') else 'live'})")
    print(f"{'='*70}")
    
    if not history:
        print("No completed quarters.")
        return
    
    total_pnl = sum(h['realized_pnl'] for h in history)
    print(f"\n{'Q#':<<4} {'Start':>12} {'End':>12} {'P&L':>12} {'Return':>8} {'Mode':>4}")
    print("-" * 70)
    for h in history:
        mode = "C" if h['compound_mode'] else "R"
        print(f"{h['quarter']:<4} "
              f"₹{h['start_capital']:>10,.0f} "
              f"₹{h['end_value']:>10,.0f} "
              f"₹{h['realized_pnl']:>10,.0f} "
              f"{h['return_pct']:>+7.2f}% "
              f"{mode:>4}")
    print("-" * 70)
    print(f"{'TOTAL':>4} {'':>12} {'':>12} ₹{total_pnl:>10,.0f}")
    
    if cfg.get('compound_mode') and history:
        total_ret = (cfg['current_base_capital'] - cfg['original_capital']) / cfg['original_capital'] * 100
        print(f"{'TOTAL RETURN:':>30} {total_ret:>+10.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()