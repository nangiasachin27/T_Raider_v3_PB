import pandas as pd
import numpy as np

def atr_breakout_strategy(df: pd.DataFrame, lookback: int = 20, atr_period: int = 14, atr_multiplier: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    if len(df) < max(lookback, atr_period):
        df['Signal'] = 0
        return df

    # Donchian Channels using Close prices for cleaner momentum entries
    df['Upper_Band'] = df['Close'].shift(1).rolling(window=lookback).max()
    df['Lower_Band'] = df['Close'].shift(1).rolling(window=lookback).min()
    
    # Accurate True Range & ATR calculation
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=atr_period).mean()
    
    signals = np.zeros(len(df))
    in_position = False
    stop_loss = 0.0
    
    closes = df['Close'].values
    upper_bands = df['Upper_Band'].values
    lower_bands = df['Lower_Band'].values
    atrs = df['ATR'].values
    
    for i in range(len(df)):
        if not in_position:
            # Entry: Close crosses above the rolling maximum Close
            if i > 0 and closes[i] > upper_bands[i] and not pd.isna(upper_bands[i]):
                in_position = True
                signals[i] = 1
                stop_loss = closes[i] - (atr_multiplier * atrs[i] if not pd.isna(atrs[i]) else closes[i] * 0.05)
        else:
            # Trailing Stop calculation
            current_stop = closes[i] - (atr_multiplier * atrs[i] if not pd.isna(atrs[i]) else closes[i] * 0.05)
            stop_loss = max(stop_loss, current_stop)
            
            # Exit: Trailing stop hit OR broken below lower channel band
            if closes[i] < stop_loss or closes[i] < lower_bands[i]:
                in_position = False
                signals[i] = -1
                
    df['Signal'] = signals
    return df