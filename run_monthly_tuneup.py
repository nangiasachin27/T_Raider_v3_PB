"""
run_monthly_tuneup.py
Executes the Auto Optimizer, generates Stock DNA profiles, and runs Research Lab analysis.
"""
import sys
import os

from auto_optimizer import optimize_hybrid_universe

# Path fix for the research module
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from research.profiler import generate_profiles
from research.analyzer import ResearchLab

if __name__ == "__main__":
    print("==================================================")
    print("🔄 STARTING MONTHLY TUNE-UP & RESEARCH CYCLE")
    print("==================================================")
    
    # 1. Run the heavy lifting (Monte Carlo & Strategy mapping)
    optimize_hybrid_universe()
    
    # 2. Build the Stock Profiles DNA Dataset
    generate_profiles()
    
    print("\n==================================================")
    print("🔬 LAUNCHING RESEARCH LAB ANALYSIS")
    print("==================================================")
    
    # 3. Analyze the results
    lab = ResearchLab()
    lab.generate_coverage_report()
    lab.profile_unsupported_behavior()
    lab.generate_suggestions()
    lab.record_historical_knowledge()
    
    print("\n✅ Monthly Tune-up Complete.")