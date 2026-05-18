import json
from pathlib import Path
from datetime import datetime

PORTFOLIO_PATH = Path("config/portfolio.json")

with open(PORTFOLIO_PATH) as f:
    portfolio = json.load(f)

holdings = portfolio.get('holdings', {})
history = portfolio.get('history', [])

migrated = 0
for ticker, holding in holdings.items():
    if isinstance(holding, int):
        # Find most recent buy
        entry_price = 0.0
        entry_date = ""
        for txn in reversed(history):
            if txn['ticker'] == ticker and txn['side'] == 'buy':
                entry_price = txn['price']
                entry_date = txn['timestamp'][:10]
                break
        
        holdings[ticker] = {
            "qty": holding,
            "entry_price": round(entry_price, 4),
            "entry_date": entry_date or datetime.now().strftime('%Y-%m-%d')
        }
        migrated += 1
        print(f"✅ Migrated {ticker}: qty={holding}, entry=₹{entry_price:.2f}")

portfolio['holdings'] = holdings

with open(PORTFOLIO_PATH, "w") as f:
    json.dump(portfolio, f, indent=2)

print(f"\n🎉 Migrated {migrated} holdings. All now have entry_price for stop-loss tracking.")