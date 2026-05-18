import json

def get_config_tickers(key="nifty_50"):
    """Loads the ticker list from config/stocks.json"""
    with open('config/stocks.json', 'r') as f:
        config = json.load(f)
    return config.get(key, [])