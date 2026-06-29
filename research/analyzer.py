"""
research/analyzer.py
Consumes optimal_params.json and stock_profiles.csv.
Responsible solely for generating research insights and reports.
"""
import json
import pandas as pd
import os
import datetime

class ResearchLab:
    def __init__(self):
        # 1. Load Optimizer Decisions
        with open('config/optimal_params.json', 'r') as f:
            self.optimal_params = json.load(f)
            
        # 2. Load Stock DNA Dataset
        try:
            self.profiles_df = pd.read_csv('config/stock_profiles.csv')
        except FileNotFoundError:
            print("⚠️ stock_profiles.csv not found! Run profiler.py first.")
            self.profiles_df = pd.DataFrame()

        # 3. Merge Strategy assignments into the DNA dataset for analysis
        if not self.profiles_df.empty:
            self.profiles_df['Strategy'] = self.profiles_df['Symbol'].map(
                lambda x: self.optimal_params.get(x, {}).get('strategy', 'NONE')
            )

    def generate_coverage_report(self):
        print("\n=== 📊 STRATEGY COVERAGE ===")
        total_stocks = len(self.optimal_params)
        if total_stocks == 0 or self.profiles_df.empty: 
            return

        strategy_counts = self.profiles_df['Strategy'].value_counts()
        for strat, count in strategy_counts.items():
            pct = (count / total_stocks) * 100
            print(f"{strat:15}: {count:3} stocks ({pct:.1f}%)")
            
        none_pct = (strategy_counts.get("NONE", 0) / total_stocks) * 100
        print(f"\nTargeting to reduce 'NONE' below 20%. Current: {none_pct:.1f}%")

    def profile_unsupported_behavior(self):
        print("\n=== 🔍 BEHAVIOUR PROFILES FOR 'NONE' STOCKS ===")
        if self.profiles_df.empty:
            return
            
        rejected_df = self.profiles_df[self.profiles_df['Strategy'] == 'NONE']
        
        if rejected_df.empty:
            print("No rejected stocks to analyze!")
            return
        
        # Count classifications directly from the pre-calculated DNA Dataset
        high_vol = len(rejected_df[rejected_df['Volatility'] == 'High'])
        choppy = len(rejected_df[(rejected_df['Trend'] == 'Sideways') | (rejected_df['ADX'] < 20)])
        illiquid = len(rejected_df[rejected_df['Liquidity'] == 'Unknown'])

        print("Market Behavior currently unsupported by T_Raider (from stock_profiles.csv):")
        print(f" - Highly Volatile (Needs wider stops / Options): {high_vol} stocks")
        print(f" - Choppy/Sideways (Needs Iron Condor / Range Bound strat): {choppy} stocks")
        print(f" - Illiquid/Missing Data: {illiquid} stocks")

    def generate_suggestions(self):
        print("\n=== 💡 FUTURE STRATEGY IDEAS ===")
        if self.profiles_df.empty: return
        
        active_strats = set(self.profiles_df['Strategy'].unique())
        print("Based on Behaviour Profiles and current system capabilities:")
        
        if "RSI_DIVERGENCE" not in active_strats:
            print(" 1. Range-Bound Strategy (e.g., RSI Divergence) for Choppy stocks.")
        else:
            print(" ✅ RSI_DIVERGENCE is active, but choppy stocks still fail.")
            print("    -> Next Step: Directional trading fails here. Consider adding an Options module (e.g., Iron Condors) to farm premium on flat stocks.")

        if "ATR_BREAKOUT" not in active_strats:
            print(" 2. High-Beta breakout strategy with ATR stops for Volatile stocks.")
        else:
            print(" ✅ ATR_BREAKOUT is active, but volatile stocks still fail.")
            print("    -> Next Step: The backtester's hard 10% stop-loss is likely choking out volatile trades before they run. Consider testing a 15% stop for high-beta assets.")

    def record_historical_knowledge(self):
        history_file = 'config/research_history.json'
        total_stocks = len(self.optimal_params)
        if total_stocks == 0 or self.profiles_df.empty: 
            return
            
        current_stats = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "total_universe": total_stocks,
            "strategies": self.profiles_df['Strategy'].value_counts().to_dict()
        }
        
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
            
        if history and history[-1]["date"] == current_stats["date"]:
            history[-1] = current_stats
        else:
            history.append(current_stats)
            
        os.makedirs('config', exist_ok=True)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
            
        print(f"\n✅ Historical knowledge updated. Currently tracking {len(history)} optimization cycles.")

if __name__ == "__main__":
    lab = ResearchLab()
    lab.generate_coverage_report()
    lab.profile_unsupported_behavior()
    lab.generate_suggestions()
    lab.record_historical_knowledge()