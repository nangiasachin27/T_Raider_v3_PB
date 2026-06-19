import pandas as pd


class SimpleBacktester:
    def __init__(self, initial_capital=100000.0, stop_loss_pct=0.10,
                 trailing_stop_pct=0.15, friction_pct=0.002):
        """
        stop_loss_pct    : Fixed stop from entry price. Protects immediately after buy.
                           Default 10%. Triggers if price drops 10% from entry.

        trailing_stop_pct: Trailing stop from the highest price reached since entry.
                           Default 15%. Activates once the trade is profitable and
                           rises with the price, locking in gains.

        How both stops work together:
            effective_stop = max(fixed_stop, trailing_stop)

        Example:
            Buy at ₹100. Fixed stop = ₹90 (10% below entry).
            Price rises to ₹130. Trailing stop = ₹110.50 (15% below ₹130).
            effective_stop is now ₹110.50 — higher than the fixed ₹90.
            If price drops to ₹110.50, trailing stop fires and we exit with ~10% gain
            instead of watching it fall all the way back to ₹90.

        friction_pct: 0.2% per trade side covers STT, brokerage, and slippage.
        """
        self.initial_capital   = initial_capital
        self.capital           = initial_capital
        self.shares_owned      = 0
        self.buy_price         = 0.0
        self.entry_date        = None
        self.peak_price        = 0.0          # tracks highest price since entry
        self.stop_loss_pct     = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.friction_pct      = friction_pct
        self.portfolio_history = []
        self.trade_log         = []
        self.trades            = []

    def _compute_effective_stop(self) -> float:
        """
        Returns the higher of the fixed stop and the trailing stop.
        The trailing stop trails the peak price reached since entry.
        The fixed stop anchors to the original entry price.

        We always use whichever is HIGHER so the stop only ever moves up,
        never down — protecting both capital and locked-in gains.
        """
        fixed_stop    = self.buy_price  * (1 - self.stop_loss_pct)
        trailing_stop = self.peak_price * (1 - self.trailing_stop_pct)
        return max(fixed_stop, trailing_stop)

    def run(self, strategy_df):
        self.capital         = self.initial_capital
        self.shares_owned    = 0
        self.buy_price       = 0.0
        self.peak_price      = 0.0
        self.entry_date      = None
        self.portfolio_history = []
        self.trade_log         = []
        self.trades            = []

        # ── Numpy array extraction (53× faster than iterrows) ─────────────────
        # pandas iterrows() boxes each row into a Series with dtype inference —
        # for a 500-row backtest called 228,096 times, that overhead dominates.
        # Extracting to numpy arrays first and indexing directly is equivalent
        # but eliminates all per-row Python object creation.
        prices  = strategy_df['Price'].to_numpy(dtype=float)
        signals = strategy_df['Signal'].to_numpy(dtype=int)
        dates   = strategy_df.index
        n       = len(prices)

        for i in range(n):
            price  = prices[i]
            signal = signals[i]
            date   = dates[i]

            # ── STOP LOSS (fixed + trailing combined) ─────────────────────
            if self.shares_owned > 0:
                prev_peak = self.peak_price
                self.peak_price = max(self.peak_price, price)

                effective_stop = self._compute_effective_stop()

                if price <= effective_stop:
                    revenue = (self.shares_owned * price) * (1 - self.friction_pct)

                    fixed_stop    = self.buy_price * (1 - self.stop_loss_pct)
                    trailing_stop = prev_peak      * (1 - self.trailing_stop_pct)
                    stop_type     = "TRAILING STOP" if trailing_stop > fixed_stop else "STOP LOSS"

                    self.trades.append({
                        'entry_price': self.buy_price,
                        'exit_price':  price * (1 - self.friction_pct),
                        'entry_date':  self.entry_date,
                        'exit_date':   date,
                    })

                    self.capital += revenue
                    self.trade_log.append({
                        'Date':    date,
                        'Type':    stop_type,
                        'Price':   price,
                        'Peak':    prev_peak,
                        'Stop_At': round(effective_stop, 2),
                        'Value':   revenue,
                    })
                    self.shares_owned = 0
                    self.buy_price    = 0.0
                    self.peak_price   = 0.0

            # ── BUY ───────────────────────────────────────────────────────
            if signal == 1 and self.shares_owned == 0:
                effective_capital = self.capital * (1 - self.friction_pct)
                self.shares_owned = int(effective_capital // price)
                self.buy_price    = price
                self.peak_price   = price
                self.entry_date   = date
                self.capital     -= (self.shares_owned * price)
                self.trade_log.append({
                    'Date':  date,
                    'Type':  'BUY',
                    'Price': price,
                })

            # ── SELL (strategy signal) ─────────────────────────────────────
            elif signal == -1 and self.shares_owned > 0:
                revenue = (self.shares_owned * price) * (1 - self.friction_pct)

                self.trades.append({
                    'entry_price': self.buy_price,
                    'exit_price':  price * (1 - self.friction_pct),
                    'entry_date':  self.entry_date,
                    'exit_date':   date,
                })

                self.capital += revenue
                self.trade_log.append({
                    'Date':  date,
                    'Type':  'SELL',
                    'Price': price,
                })
                self.shares_owned = 0
                self.buy_price    = 0.0
                self.peak_price   = 0.0

            current_value = self.capital + (self.shares_owned * price)
            self.portfolio_history.append(current_value)

        while len(self.portfolio_history) < len(strategy_df):
            self.portfolio_history.append(self.portfolio_history[-1])

        strategy_df['Portfolio_Value'] = self.portfolio_history
        return strategy_df, pd.DataFrame(self.trade_log)

    def get_metrics(self, strategy_df):
        final_value = strategy_df['Portfolio_Value'].iloc[-1]
        days  = (strategy_df.index[-1] - strategy_df.index[0]).days
        years = days / 365.25
        annualized_return = (
            ((final_value / self.initial_capital) ** (1 / years)) - 1
        ) * 100 if years > 0 else 0

        strategy_df['Peak']     = strategy_df['Portfolio_Value'].cummax()
        strategy_df['Drawdown'] = (
            (strategy_df['Portfolio_Value'] - strategy_df['Peak'])
            / strategy_df['Peak']
        )
        max_drawdown = strategy_df['Drawdown'].min() * 100

        # Count stop types from trade log
        tl             = pd.DataFrame(self.trade_log)
        trailing_exits = len(tl[tl['Type'] == 'TRAILING STOP']) if not tl.empty else 0
        fixed_exits    = len(tl[tl['Type'] == 'STOP LOSS'])     if not tl.empty else 0

        return {
            'Net Final Value':     f"{final_value:,.2f}",
            'Post-Tax Annualized': f"{annualized_return:.2f}%",
            'Max Drawdown':        f"{max_drawdown:.2f}%",
            'Trailing Stop Exits': trailing_exits,
            'Fixed Stop Exits':    fixed_exits,
            'Friction Accounted':  "0.2% per trade",
        }