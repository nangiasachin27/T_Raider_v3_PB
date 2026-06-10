import json

def get_config_tickers():
    """Loads the full ticker list from all categories in config/stocks.json"""
    with open('config/stocks.json', 'r') as f:
        config = json.load(f)
        
    all_tickers = []
    for value in config.values():
        if isinstance(value, list):
            all_tickers.extend(value)
            
    # Deduplicate while preserving order
    seen = set()
    unique_tickers = [t for t in all_tickers if not (t in seen or seen.add(t))]
    
    return unique_tickers