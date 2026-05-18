import json
import yfinance as yf
import pandas as pd
from pathlib import Path

def get_sector_ranks(lookback_days=60, map_path="config/dynamic_sector_map.json"):
    """
    Ranks only the sectors that actually exist in your current stocks.json universe.
    """
    if not Path(map_path).exists():
        print("⚠️ Sector map not found. Run sector_mapper.py first.")
        return {}

    with open(map_path, "r") as f:
        sector_map = json.load(f)

    # Find all unique NSE indices needed for your current universe
    unique_indices = set(data["nse_index"] for data in sector_map.values() if data["nse_index"] != "UNKNOWN")
    tickers_to_download = list(unique_indices) + ["^NSEI"] # Always include Nifty 50 for the baseline
    
    # Download data
    data = yf.download(tickers_to_download, period="3mo", progress=False)['Close']
    
    if data.empty or "^NSEI" not in data.columns:
        return {}

    performance = {}
    for ticker in tickers_to_download:
        if ticker in data.columns:
            series = data[ticker].dropna()
            if len(series) > lookback_days // 2:
                start_price = series.iloc[-lookback_days] if len(series) >= lookback_days else series.iloc[0]
                end_price = series.iloc[-1]
                performance[ticker] = ((end_price - start_price) / start_price) * 100

    nifty_return = performance.pop("^NSEI", 0)
    
    # Calculate Relative Strength
    rs_ranks = []
    for ticker, ret in performance.items():
        rs_ranks.append({
            "sector_index": ticker,
            "rs_score": ret - nifty_return
        })
        
    # Sort and create dictionary
    df_ranks = pd.DataFrame(rs_ranks).sort_values(by="rs_score", ascending=False)
    
    rank_dict = {}
    for i, row in enumerate(df_ranks.itertuples()):
        rank_dict[row.sector_index] = {
            "rank": i + 1,
            "rs_score": row.rs_score,
            "is_outperforming": row.rs_score > 0
        }
        
    return rank_dict