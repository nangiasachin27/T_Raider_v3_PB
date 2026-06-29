"""
run_monthly_tuneup.py
Executes the Auto Optimizer and immediately pipes the output to the Research Lab.
"""
import sys
import os

from auto_optimizer import optimize_hybrid_universe

# Path fix for the research module
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from research.analyzer import ResearchLab

if __name__ == "__main__":
    print("==================================================")
    print("🔄 STARTING MONTHLY TUNE-UP & RESEARCH CYCLE")
    print("==================================================")
    
    # 1. Run the heavy lifting (Monte Carlo & Strategy mapping)
    optimize_hybrid_universe()
    
    print("\n==================================================")
    print("🔬 LAUNCHING RESEARCH LAB ANALYSIS")
    print("==================================================")
    
    # 2. Analyze the results
    lab = ResearchLab()
    lab.generate_coverage_report()
    lab.profile_unsupported_behavior()
    lab.generate_suggestions()
    lab.record_historical_knowledge()
    
    print("\n✅ Monthly Tune-up Complete.")