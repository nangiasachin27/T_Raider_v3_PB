import pandas as pd

def apply_golden_cross_strategy(price_series):
    """
    Applies an upgraded Trend Pullback strategy to prevent 'lock-outs'.
    """
    df = pd.DataFrame(price_series)
    df.columns = ['Price']
    
    # 1. Calculate Exponential Moving Averages (EMAs)
    df['EMA_50'] = df['Price'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Price'].ewm(span=200, adjust=False).mean()
    
    df['Signal'] = 0 
    
    # 2. The Macro Filter: Is the overall trend up?
    is_uptrend = df['EMA_50'] > df['EMA_200']
    
    # 3. Identify the Entry (Buy Signal)
    # Condition: We are in an uptrend, AND the price just crossed ABOVE the 50 EMA today
    buy_condition = is_uptrend & (df['Price'] > df['EMA_50']) & (df['Price'].shift(1) <= df['EMA_50'].shift(1))
    
    # 4. Identify the Exit (Sell Signal)
    # Condition: The price just crossed BELOW the 50 EMA today
    sell_condition = is_uptrend & (df['Price'] < df['EMA_50']) & (df['Price'].shift(1) >= df['EMA_50'].shift(1))
    
    df.loc[buy_condition, 'Signal'] = 1
    df.loc[sell_condition, 'Signal'] = -1
    
    return df

if __name__ == "__main__":
    print("Trend follower module ready.")