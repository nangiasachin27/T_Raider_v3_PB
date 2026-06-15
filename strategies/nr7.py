import pandas as pd
import numpy as np

def execute_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    NR7 (Narrowest Range 7) Volatility Contraction Breakout.
    Flags when daily range (High - Low) is the lowest in 7 days.
    Buys when the next bar breaks above the NR7 High.
    Required columns: 'High', 'Low', 'Close'
    """
    df = df.copy()
    df['Signal'] = 0
    
    if len(df) < 8:
        return df
        
    # 1. Calculate Daily Range
    df['Range'] = df['High'] - df['Low']
    
    # 2. Identify if today is the narrowest range in the last 7 sessions
    df['Is_NR7'] = df['Range'] == df['Range'].rolling(window=7).min()
    
    # 3. Establish breakout levels from the previous session if it was an NR7
    df['Breakout_High'] = np.where(df['Is_NR7'].shift(1), df['High'].shift(1), np.nan)
    df['Breakout_Low'] = np.where(df['Is_NR7'].shift(1), df['Low'].shift(1), np.nan)
    
    # Forward fill levels during the active breakout window (e.g., 3 days max)
    df['Breakout_High'] = df['Breakout_High'].ffill(limit=3)
    df['Breakout_Low'] = df['Breakout_Low'].ffill(limit=3)
    
    # 4. Fire Entry/Exit Signals
    buy_cond = (df['Close'] > df['Breakout_High']) & (df['Close'].shift(1) <= df['Breakout_High'])
    sell_cond = (df['Close'] < df['Breakout_Low']) | (df['Close'] < df['Close'].rolling(window=10).mean())
    
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    
    return df