import pandas as pd

def apply_bollinger_strategy(price_series, window=20, num_std=2):
    """
    Bollinger Band Mean-Reversion Strategy:
    Upper Band = 20-day MA + 2 standard deviations
    Lower Band = 20-day MA - 2 standard deviations

    Signal logic (CROSSOVER, not position):
        BUY  (+1) : Price crosses BELOW the Lower Band today
                    (was at or above Lower Band yesterday, below today)
        SELL (-1) : Price crosses ABOVE the Upper Band today
                    (was at or below Upper Band yesterday, above today)
        HOLD ( 0) : No band breach on this bar

    FIX: The original code set Signal = 1 on every bar where
    price < Lower Band and Signal = -1 on every bar where
    price > Upper Band. During a sharp sell-off, price can stay
    below the lower band for 5-10 consecutive days, firing 5-10
    BUY signals for the same move. The backtester ignores the
    duplicates (position guard), but the optimizer inflates the
    trade count and computes a distorted win rate and expected
    return for any stock assigned this strategy. The screener
    also treats a week-old band breach as a fresh signal today.

    The fix uses .shift(1) to detect the first bar of the breach
    only — the same crossover pattern used by RSI and MACD.

    Note on mean-reversion intent:
        Bollinger mean-reversion assumes the breach is temporary
        and price will revert to the MA. Firing only on the entry
        bar is correct — you want the signal at the moment of
        the breach, not every day you're stuck in a losing trade
        waiting for reversion.
    """
    df = pd.DataFrame(price_series)
    df.columns = ['Price']

    # ── Bands ─────────────────────────────────────────────────────────────
    df['MA']    = df['Price'].rolling(window=window).mean()
    df['STD']   = df['Price'].rolling(window=window).std()
    df['Upper'] = df['MA'] + (num_std * df['STD'])
    df['Lower'] = df['MA'] - (num_std * df['STD'])

    # ── Crossover signals ─────────────────────────────────────────────────
    price_today = df['Price']
    price_prev  = df['Price'].shift(1)
    lower_today = df['Lower']
    lower_prev  = df['Lower'].shift(1)
    upper_today = df['Upper']
    upper_prev  = df['Upper'].shift(1)

    df['Signal'] = 0

    # BUY: price was at or above Lower Band yesterday, crossed below today
    buy_condition  = (price_today < lower_today) & (price_prev >= lower_prev)

    # SELL: price was at or below Upper Band yesterday, crossed above today
    sell_condition = (price_today > upper_today) & (price_prev <= upper_prev)

    df.loc[buy_condition,  'Signal'] =  1
    df.loc[sell_condition, 'Signal'] = -1

    return df