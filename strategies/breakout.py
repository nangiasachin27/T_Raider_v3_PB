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
    # Buy: Price closes above previous window high
    df.loc[df['Price'] > df['Upper'], 'Signal'] = 1
    # Sell: Price closes below previous window low
    df.loc[df['Price'] < df['Lower'], 'Signal'] = -1
    
    return df