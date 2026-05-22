import pandas as pd

def apply_stretch_strategy(price_series, window=20, threshold=0.05):
    """
    Mean Reversion (Stretch):
    Buy when price is 'threshold'% BELOW the N-day Moving Average.
    Sell when price is 'threshold'% ABOVE the N-day Moving Average.
    """
    df = pd.DataFrame(price_series)
    df.columns = ['Price']
    
    # Calculate the Mean (Moving Average)
    df['MA'] = df['Price'].rolling(window=window).mean()
    
    # Calculate the % Deviation from Mean
    df['Deviation'] = (df['Price'] - df['MA']) / df['MA']
    
    df['Signal'] = 0

    # ── Crossover signals (FIX: only fire on the first bar of the breach) ──
    dev_today = df['Deviation']
    dev_prev = df['Deviation'].shift(1)

    # BUY: Price was at or above the threshold yesterday, crossed below today
    buy_condition = (dev_today < -threshold) & (dev_prev >= -threshold)

    # SELL: Price was at or below the threshold yesterday, crossed above today
    sell_condition = (dev_today > threshold) & (dev_prev <= threshold)

    df.loc[buy_condition, 'Signal'] = 1
    df.loc[sell_condition, 'Signal'] = -1

    return df