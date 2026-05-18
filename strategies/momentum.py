import pandas as pd

def apply_macd_strategy(price_series, fast=12, slow=26, signal=9):
    """
    Standard MACD Crossover Strategy:
    MACD Line  = fast EMA - slow EMA  (default 12 - 26)
    Signal Line = 9-period EMA of MACD Line
    Histogram   = MACD Line - Signal Line

    Signal logic (CROSSOVER, not position):
        BUY  (+1) : MACD Line crosses ABOVE Signal Line today
                    (was below or equal yesterday, above today)
        SELL (-1) : MACD Line crosses BELOW Signal Line today
                    (was above or equal yesterday, below today)
        HOLD ( 0) : No crossover on this bar

    FIX: The original code set Signal = 1 on every bar where
    MACD > Signal Line, and Signal = -1 on every bar where
    MACD < Signal Line. That means a stock trending up for 30
    days fires 30 consecutive BUY signals. The backtester only
    acts on the first one (shares_owned > 0 guard), but the
    optimizer sees inflated signal counts and the screener can
    misreport a stale multi-week condition as a fresh entry.

    The fix uses .shift(1) to compare today's position against
    yesterday's, triggering only on the actual crossover bar —
    exactly matching the design of RSI and trend_follower.
    """
    df = pd.DataFrame(price_series)
    df.columns = ['Price']

    # ── MACD Components ───────────────────────────────────────────────────
    ema_fast = df['Price'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Price'].ewm(span=slow, adjust=False).mean()

    df['MACD_Line']   = ema_fast - ema_slow
    df['Signal_Line'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
    df['Histogram']   = df['MACD_Line'] - df['Signal_Line']

    # ── Crossover signals ─────────────────────────────────────────────────
    macd_today = df['MACD_Line']
    macd_prev  = df['MACD_Line'].shift(1)
    sig_today  = df['Signal_Line']
    sig_prev   = df['Signal_Line'].shift(1)

    df['Signal'] = 0

    # BUY: MACD was at or below Signal Line yesterday, crossed above today
    buy_condition  = (macd_today > sig_today) & (macd_prev <= sig_prev)

    # SELL: MACD was at or above Signal Line yesterday, crossed below today
    sell_condition = (macd_today < sig_today) & (macd_prev >= sig_prev)

    df.loc[buy_condition,  'Signal'] =  1
    df.loc[sell_condition, 'Signal'] = -1

    return df