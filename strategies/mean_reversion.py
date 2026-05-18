import pandas as pd
import numpy as np

def apply_rsi_strategy(price_series, window=14, buy=30, sell=70):
    """
    Applies a Mean Reversion strategy using the Relative Strength Index (RSI).
    Standardized parameter names: 'buy' and 'sell' to match the Optimizer.
    """
    df = pd.DataFrame(price_series)
    df.columns = ['Price']
    
    # 1. Calculate the daily price change
    delta = df['Price'].diff()
    
    # 2. Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # 3. Calculate smoothing (Wilder's Smoothing)
    ema_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    ema_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    
    # 4. Calculate RSI (with safety for zero-division)
    rs = ema_gain / ema_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50) # Neutral if no movement
    
    # 5. Generate Signals
    df['Signal'] = 0
    
    # BUY: RSI enters oversold territory
    buy_condition = (df['RSI'] < buy) & (df['RSI'].shift(1) >= buy)
    
    # SELL: RSI enters overbought territory
    sell_condition = (df['RSI'] > sell) & (df['RSI'].shift(1) <= sell)
    
    df.loc[buy_condition, 'Signal'] = 1
    df.loc[sell_condition, 'Signal'] = -1
    
    return df

if __name__ == "__main__":
    print("Mean Reversion (RSI) module standardized for T_Raider Optimizer.")