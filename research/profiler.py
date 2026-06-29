"""
research/profiler.py
Generates the "Stock DNA" dataset (stock_profiles.csv).
Responsible solely for describing stock behaviour (Trend, Volatility, Liquidity).
"""
import pandas as pd
import numpy as np
import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_config_tickers
from ingestion.data_ingestion import fetch_historical_data, get_stock_data

def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculates behavioral metrics for a single stock."""
    df = df.dropna(subset=['Close', 'High', 'Low', 'Volume']).copy()
    
    if df.empty or len(df) < 200:
        return None
        
    close = df['Close'].iloc[-1]
    
    # 1. ATR %
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    atr_pct = (atr / close) * 100

    # 2. Trend Divergence (50 SMA vs 200 SMA)
    sma_50 = df['Close'].rolling(50).mean().iloc[-1]
    sma_200 = df['Close'].rolling(200).mean().iloc[-1]
    trend_div = ((sma_50 - sma_200) / sma_200) * 100 if sma_200 else 0

    # 3. Average Volume
    avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
    
    # 4. ADX (Average Directional Index) - Simplified approximation
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0))
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0))
    atr_series = tr.rolling(14).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_series)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_series)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/14, adjust=False).mean().iloc[-1]

    # 5. Gap Frequency (% of days with >1% opening gap)
    gaps = (df['Open'] - df['Close'].shift(1)).abs() / df['Close'].shift(1) * 100
    gap_freq = (gaps > 1.0).sum() / len(gaps) * 100

    # 6. Classifications
    volatility = "High" if atr_pct > 2.0 else ("Medium" if atr_pct > 1.0 else "Low")
    trend = "Uptrend" if trend_div > 5.0 else ("Downtrend" if trend_div < -5.0 else "Sideways")
    liquidity = "Excellent" if avg_vol > 1000000 else ("Good" if avg_vol > 100000 else "Poor")

    return {
        "ATR_%": round(atr_pct, 2),
        "ADX": round(adx, 2),
        "Trend_Divergence_%": round(trend_div, 2),
        "Gap_Freq_%": round(gap_freq, 2),
        "Avg_Volume": int(avg_vol),
        "Volatility": volatility,
        "Trend": trend,
        "Liquidity": liquidity
    }

def generate_profiles():
    print("==================================================")
    print("🧬 T_RAIDER PROFILER: GENERATING STOCK DNA")
    print("==================================================")
    
    tickers = get_config_tickers()
    print(f"Fetching 1y data to profile {len(tickers)} stocks...")
    market_data = fetch_historical_data(tickers, period="1y")
    
    profiles = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for ticker in tickers:
        df = get_stock_data(market_data, ticker)
        metrics = calculate_metrics(df)
        
        if metrics:
            profile = {"Symbol": ticker}
            profile.update(metrics)
            profile["Generated_At"] = timestamp
            profiles.append(profile)
        else:
            print(f"⚠️ Insufficient data to profile {ticker}")
            profiles.append({
                "Symbol": ticker,
                "ATR_%": None, "ADX": None, "Trend_Divergence_%": None,
                "Gap_Freq_%": None, "Avg_Volume": None,
                "Volatility": "Unknown", "Trend": "Unknown", "Liquidity": "Unknown",
                "Generated_At": timestamp
            })
    
    profiles_df = pd.DataFrame(profiles)
    os.makedirs("config", exist_ok=True)
    profiles_df.to_csv("config/stock_profiles.csv", index=False)
    
    print(f"✅ Generated behavioural profiles for {len(profiles)} stocks.")
    print("💾 Saved to config/stock_profiles.csv")
    return profiles_df

if __name__ == "__main__":
    generate_profiles()