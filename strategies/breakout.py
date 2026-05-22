import pandas as pd

def apply_breakout_strategy(price_series, window=20):
    """
    Donchian Channel Breakout:
    Buy when price breaks the Highest High of the last N days.
    Sell when price breaks the Lowest Low of the last N days.
    """
    df = pd.DataFrame(price_series)
    df.columns = ['Price']
    
    # Resistance (Upper Band) and Support (Lower Band)
    # We use .shift(1) because we want to break the PREVIOUS days' high
    df['Upper'] = df['Price'].rolling(window=window).max().shift(1)
    df['Lower'] = df['Price'].rolling(window=window).min().shift(1)
    
    df['Signal'] = 0

    price_today = df['Price']
    price_prev = df['Price'].shift(1)
    upper_today = df['Upper']
    upper_prev = df['Upper'].shift(1)
    lower_today = df['Lower']
    lower_prev = df['Lower'].shift(1)

    # BUY: Price was at or below the upper band yesterday, broke above today
    buy_condition = (price_today > upper_today) & (price_prev <= upper_prev)

    # SELL: Price was at or above the lower band yesterday, broke below today
    sell_condition = (price_today < lower_today) & (price_prev >= lower_prev)

    df.loc[buy_condition, 'Signal'] = 1
    df.loc[sell_condition, 'Signal'] = -1
    
    return df