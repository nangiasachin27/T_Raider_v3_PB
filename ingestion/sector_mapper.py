import json
import yfinance as yf
from pathlib import Path
from ingestion.nse_constituents import load_universe

# We map Yahoo Finance's standard global sector names to NSE specific indices
# NO stock tickers are hardcoded here, only sector categories.
YF_TO_NSE_INDEX = {
    "Information Technology": "^CNXIT",
    "Financial Services": "^CNXFIN",     # Financial Services / Banks
    "Energy": "^CNXENERGY",
    "Consumer Cyclical": "^CNXAUTO",     # In India, this is heavily Auto
    "Consumer Defensive": "^CNXFMCG",
    "Healthcare": "^CNXPHARMA",
    "Basic Materials": "^CNXMETAL",
    "Industrials": "^CNXINFRA",
    "Real Estate": "^CNXREALTY",
    "Communication Services": "^CNXMEDIA",
    "Utilities": "^CNXENERGY"            # Grouping power/utilities with Energy
}

def generate_sector_map(output_path="config/dynamic_sector_map.json"):
    """
    Reads stocks.json, dynamically fetches their sectors, and maps them to NSE Indices.
    """
    print("🔍 Reading stocks from stocks.json...")
    tickers = load_universe("config/stocks.json")
    
    sector_map = {}
    print(f"🌐 Fetching sector data for {len(tickers)} stocks. This may take a minute...")
    
    for ticker in tickers:
        try:
            # fast_info doesn't have sector, so we use .info
            stock_info = yf.Ticker(ticker).info
            yf_sector = stock_info.get("sector", "Unknown")
            
            # Map it to our NSE indices
            nse_index = YF_TO_NSE_INDEX.get(yf_sector, "UNKNOWN")
            
            sector_map[ticker] = {
                "yf_sector": yf_sector,
                "nse_index": nse_index
            }
            print(f"  ✓ {ticker:15} -> {yf_sector} ({nse_index})")
            
        except Exception as e:
            print(f"  ❌ Failed to fetch {ticker}: {e}")
            sector_map[ticker] = {"yf_sector": "Unknown", "nse_index": "UNKNOWN"}
            
    # Save the mapping to a file so the screener can load it instantly
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sector_map, f, indent=4)
        
    print(f"\n✅ Dynamic Sector Map saved to {output_path}")
    return sector_map

if __name__ == "__main__":
    generate_sector_map()