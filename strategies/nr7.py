import pandas as pd
import numpy as np

def execute_strategy(df: pd.DataFrame, lookback: int = 7, breakout_window: int = 3) -> pd.DataFrame:
    """
    NR-N Volatility Contraction Breakout.
    Flags when daily range (High - Low) is the lowest in `lookback` days.
    Buys when the next bar breaks above the NR high.
    Required columns: 'High', 'Low', 'Close'

    Args:
        lookback        : Number of sessions to look back for range contraction (default 7 = NR7).
                          Smaller values (4/5) fire more frequently on tighter squeezes;
                          larger values (10) catch rarer but more decisive breakouts.
        breakout_window : Max sessions to keep the breakout level active via forward-fill (default 3).
                          Tighter windows require the breakout to happen sooner after the squeeze.
    """
    df = df.copy()
    df['Signal'] = 0

    if len(df) < lookback + 1:
        return df

    # 1. Calculate Daily Range
    df['Range'] = df['High'] - df['Low']

    # 2. Identify if today is the narrowest range in the last N sessions
    df['Is_NR'] = df['Range'] == df['Range'].rolling(window=lookback).min()

    # 3. Establish breakout levels from the previous session if it was an NR day
    df['Breakout_High'] = np.where(df['Is_NR'].shift(1), df['High'].shift(1), np.nan)
    df['Breakout_Low']  = np.where(df['Is_NR'].shift(1), df['Low'].shift(1),  np.nan)

    # Forward fill levels during the active breakout window
    df['Breakout_High'] = df['Breakout_High'].ffill(limit=breakout_window)
    df['Breakout_Low']  = df['Breakout_Low'].ffill(limit=breakout_window)

    # 4. Fire Entry/Exit Signals
    buy_cond  = (df['Close'] > df['Breakout_High']) & (df['Close'].shift(1) <= df['Breakout_High'])
    sell_cond = (df['Close'] < df['Breakout_Low'])  | (df['Close'] < df['Close'].rolling(window=10).mean())

    df.loc[buy_cond,  'Signal'] =  1
    df.loc[sell_cond, 'Signal'] = -1

    return df