"""
execution/instrument_mapper.py
──────────────────────────────
Maps NSE ticker symbols to Upstox instrument tokens.
"""

import json
import csv
import requests
from pathlib import Path
from typing import Optional

CACHE_PATH = Path("config/upstox_instrument_cache.json")
MASTER_CSV_URL = "https://assets.upstox.com/market-data/instruments/exchange/NSE.csv"


class InstrumentMapper:
    def __init__(self, cache_path: Path = CACHE_PATH):
        self.cache_path = cache_path
        self._cache = self._load_cache()
    
    def get_token(self, symbol: str) -> Optional[str]:
        token = self._cache.get(symbol.upper())
        if not token:
            self._refresh_cache()
            token = self._cache.get(symbol.upper())
        return token
    
    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}
    
    def _save_cache(self) -> None:
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)
    
    def _refresh_cache(self) -> None:
        try:
            print("📥 Refreshing Upstox instrument cache...")
            resp = requests.get(MASTER_CSV_URL, timeout=30)
            resp.raise_for_status()
            
            decoded = resp.content.decode('utf-8').splitlines()
            reader = csv.DictReader(decoded)
            
            new_cache = {}
            for row in reader:
                if row.get('exchange') == 'NSE' and row.get('segment') == 'NSE_EQ':
                    sym = row.get('tradingsymbol', '').strip().upper()
                    key = row.get('instrument_key', '').strip()
                    if sym and key:
                        new_cache[sym] = key
            
            self._cache = new_cache
            self._save_cache()
            print(f"✅ Cached {len(new_cache)} NSE instruments.")
            
        except Exception as e:
            print(f"⚠️ Failed to refresh cache: {e}")


if __name__ == "__main__":
    mapper = InstrumentMapper()
    mapper._refresh_cache()
    test = mapper.get_token("RELIANCE")
    print(f"RELIANCE token: {test}")