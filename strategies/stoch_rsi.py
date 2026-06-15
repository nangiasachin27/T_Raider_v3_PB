import pandas as pd
import numpy as np

def execute_strategy(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14, 
                     k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    """
    Stochastic RSI Strategy. Overcomes standard RSI friction in strong trends.
    Buys when Fast K crosses above Slow D in the oversold territory (< 20).
    Required columns: 'Close'
    """
    df = df.copy()
    df['Signal'] = 0
    
    if len(df) < (rsi_period + stoch_period):
        return df

    # 1. Calculate Standard RSI
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean()
    
    rs = avg_gain / np.where(avg_loss == 0, 0.00001, avg_loss)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. Calculate Stochastic applied over the RSI series
    min_rsi = df['RSI'].rolling(window=stoch_period).min()
    max_rsi = df['RSI'].rolling(window=stoch_period).max()
    
    df['StochRSI'] = (df['RSI'] - min_rsi) / np.where((max_rsi - min_rsi) == 0, 0.00001, max_rsi - min_rsi)
    
    # Smoothed Lines (%K and %D)
    df['K'] = df['StochRSI'].rolling(window=k_smooth).mean() * 100
    df['D'] = df['K'].rolling(window=d_smooth).mean() * 100
    
    # 3. Generate Signals on Overlap/Crossover
    df['Prev_K'] = df['K'].shift(1)
    df['Prev_D'] = df['D'].shift(1)
    
    buy_cond = (df['K'] > df['D']) & (df['Prev_K'] <= df['Prev_D']) & (df['K'] < 20)
    sell_cond = (df['K'] < df['D']) & (df['Prev_K'] >= df['Prev_D']) & (df['K'] > 80)
    
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    
    return df