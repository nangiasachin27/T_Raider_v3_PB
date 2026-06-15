import pandas as pd
import numpy as np

def execute_strategy(df: pd.DataFrame, ema_period: int = 20) -> pd.DataFrame:
    """
    On-Balance Volume (OBV) Momentum Strategy.
    Buys when OBV crosses above its EMA. Sells on cross below.
    Required columns: 'Close', 'Volume'
    """
    df = df.copy()
    df['Signal'] = 0
    
    if len(df) < max(2, ema_period):
        return df

    # 1. Calculate OBV
    close_diff = df['Close'].diff()
    obv = np.where(close_diff > 0, df['Volume'], 
                   np.where(close_diff < 0, -df['Volume'], 0))
    df['OBV'] = np.cumsum(obv)
    
    # 2. Calculate OBV Signal Line
    df['OBV_EMA'] = df['OBV'].ewm(span=ema_period, adjust=False).mean()
    
    # 3. Generate Signals on Crossover
    df['Prev_OBV'] = df['OBV'].shift(1)
    df['Prev_OBV_EMA'] = df['OBV_EMA'].shift(1)
    
    buy_cond = (df['OBV'] > df['OBV_EMA']) & (df['Prev_OBV'] <= df['Prev_OBV_EMA'])
    sell_cond = (df['OBV'] < df['OBV_EMA']) & (df['Prev_OBV'] >= df['Prev_OBV_EMA'])
    
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    
    return df