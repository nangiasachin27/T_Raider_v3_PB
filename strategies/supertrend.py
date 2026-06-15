import pandas as pd
import numpy as np

def execute_strategy(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    SuperTrend Volatility-Adjusted Trailing Trend Strategy.
    Required columns: 'High', 'Low', 'Close'
    """
    df = df.copy()
    df['Signal'] = 0
    
    if len(df) < period:
        return df

    # 1. Calculate ATR
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift(1))
    low_cp = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # 2. Calculate Basic Bands (FIX: Added .to_numpy() for integer indexing in the loop)
    hl2 = (df['High'] + df['Low']) / 2
    basic_upper = (hl2 + (multiplier * atr)).to_numpy()
    basic_lower = (hl2 - (multiplier * atr)).to_numpy()

    # 3. Calculate Final Bands (Iterative path lock)
    final_upper = np.zeros(len(df))
    final_lower = np.zeros(len(df))
    trend = np.zeros(len(df))
    close = df['Close'].to_numpy()

    for i in range(1, len(df)):
        # Final Upper Band logic
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        # Final Lower Band logic
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        # Determine Trend Direction
        if close[i] > final_upper[i]:
            trend[i] = 1
        elif close[i] < final_lower[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    df['Trend'] = trend
    
    # 4. Generate Crossover Signal Changes
    df['Prev_Trend'] = df['Trend'].shift(1)
    df.loc[(df['Trend'] == 1) & (df['Prev_Trend'] == -1), 'Signal'] = 1
    df.loc[(df['Trend'] == -1) & (df['Prev_Trend'] == 1), 'Signal'] = -1

    return df