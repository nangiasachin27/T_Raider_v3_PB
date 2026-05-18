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
    # Buy: Price is 5% (default) below the MA
    df.loc[df['Deviation'] < -threshold, 'Signal'] = 1
    # Sell: Price is 5% above the MA
    df.loc[df['Deviation'] > threshold, 'Signal'] = -1
    
    return df