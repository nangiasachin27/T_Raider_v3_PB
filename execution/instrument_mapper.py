"""
execution/instrument_mapper.py
──────────────────────────────
Maps NSE ticker symbols to Upstox instrument tokens.

FIXED: Updated to use new Upstox JSON instrument master URLs.
Upstox deprecated CSV format - now using JSON/GZ endpoints.
"""

import json
import gzip
import requests
from pathlib import Path
from typing import Optional

CACHE_PATH = Path("config/upstox_instrument_cache.json")

# NEW Upstox instrument master URLs (JSON format, replaces deprecated CSV)
# Source: https://upstox.com/developer/api-documentation/instruments/
NSE_JSON_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
COMPLETE_JSON_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"

# Fallback: static CSV (deprecated but kept as last resort)
MASTER_CSV_URL = "https://assets.upstox.com/market-data/instruments/exchange/NSE.csv"


class InstrumentMapper:
    def __init__(self, cache_path: Path = CACHE_PATH):
        self.cache_path = cache_path
        self._cache = self._load_cache()

    def get_token(self, symbol: str) -> Optional[str]:
        """Get instrument token for a given NSE symbol (e.g., 'RELIANCE.NS')."""
        # Handle .NS suffix
        clean_symbol = symbol.upper().replace('.NS', '')

        token = self._cache.get(clean_symbol)
        if not token:
            self._refresh_cache()
            token = self._cache.get(clean_symbol)
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
        """Refresh cache using new JSON endpoint (primary) with CSV fallback."""

        # Try JSON first (new format)
        if self._refresh_from_json():
            return

        # Fallback to CSV (old format, may also fail with 403)
        print("   JSON failed, trying CSV fallback...")
        self._refresh_from_csv()

    def _refresh_from_json(self) -> bool:
        """Fetch instrument master from new JSON/GZ endpoint."""
        try:
            print("📥 Refreshing Upstox instrument cache (JSON format)...")

            # Download NSE-specific JSON
            resp = requests.get(NSE_JSON_URL, timeout=30, headers={
                'User-Agent': 'T_Raider/3.0'
            })
            resp.raise_for_status()

            # Decompress gzip
            raw_data = gzip.decompress(resp.content)
            instruments = json.loads(raw_data.decode('utf-8'))

            new_cache = {}
            for instrument in instruments:
                # Filter for NSE equity segment only
                segment = instrument.get('segment', '')
                instrument_type = instrument.get('instrument_type', '')

                if segment == 'NSE_EQ' and instrument_type == 'EQ':
                    sym = instrument.get('trading_symbol', '').strip().upper()
                    key = instrument.get('instrument_key', '').strip()
                    if sym and key:
                        new_cache[sym] = key

            self._cache = new_cache
            self._save_cache()
            print(f"✅ Cached {len(new_cache)} NSE equity instruments from JSON.")
            return True

        except Exception as e:
            print(f"⚠️ JSON refresh failed: {e}")
            return False

    def _refresh_from_csv(self) -> None:
        """Fallback: Fetch from CSV (deprecated, may return 403)."""
        try:
            print("📥 Refreshing Upstox instrument cache (CSV fallback)...")
            import csv

            resp = requests.get(MASTER_CSV_URL, timeout=30, headers={
                'User-Agent': 'T_Raider/3.0'
            })
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
            print(f"✅ Cached {len(new_cache)} NSE instruments from CSV.")

        except Exception as e:
            print(f"⚠️ CSV fallback also failed: {e}")
            print("   Instrument cache remains unchanged.")


if __name__ == "__main__":
    mapper = InstrumentMapper()
    mapper._refresh_cache()

    # Test lookups
    test_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "NIFTY50"]
    print("\n🔍 Testing symbol lookups:")
    for sym in test_symbols:
        token = mapper.get_token(sym)
        status = "✅" if token else "❌"
        print(f"   {status} {sym:15} → {token}")