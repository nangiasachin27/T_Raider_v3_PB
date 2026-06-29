import pandas as pd
import numpy as np

def rsi_divergence_strategy(df, rsi_period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df, columns=['Close'])
    else:
        df = df.copy()
        if 'Close' not in df.columns and 'Price' in df.columns:
            df['Close'] = df['Price']
            
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Signal'] = 0
    
    # CROSSOVER logic (fires only once per event)
    buy_cond = (df['RSI'] < oversold) & (df['RSI'].shift(1) >= oversold)
    sell_cond = (df['RSI'] > overbought) & (df['RSI'].shift(1) <= overbought)
    
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    
    return df