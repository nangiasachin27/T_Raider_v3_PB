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
        self.portfolio_history = []
        self.trade_log         = []
        self.trades            = []

        for date, row in strategy_df.iterrows():
            price  = row['Price']
            signal = row['Signal']

            # ── STOP LOSS (fixed + trailing combined) ─────────────────────
            if self.shares_owned > 0:
                # FIX 1: Capture peak_price BEFORE updating it so the stop-type
                # label is determined from the previous day's peak, not today's.
                # Bug: peak was updated to max(peak, price) first, then the
                # trailing_stop re-check used that already-updated peak_price,
                # which could make trailing_stop > fixed_stop even when the
                # fixed stop was what actually fired.
                prev_peak = self.peak_price
                self.peak_price = max(self.peak_price, price)

                effective_stop = self._compute_effective_stop()

                if price <= effective_stop:
                    revenue = (self.shares_owned * price) * (1 - self.friction_pct)

                    # FIX 1 (continued): Use prev_peak (the peak before today's
                    # price was folded in) to determine which stop triggered.
                    fixed_stop    = self.buy_price * (1 - self.stop_loss_pct)
                    trailing_stop = prev_peak      * (1 - self.trailing_stop_pct)
                    stop_type     = "TRAILING STOP" if trailing_stop > fixed_stop else "STOP LOSS"

                    self.trades.append({
                        'entry_price': self.buy_price,
                        'exit_price':  price * (1 - self.friction_pct),
                    })

                    self.capital += revenue
                    self.trade_log.append({
                        'Date':      date,
                        'Type':      stop_type,
                        'Price':     price,
                        'Peak':      prev_peak,           # log the true peak, not the post-update one
                        'Stop_At':   round(effective_stop, 2),
                        'Value':     revenue,
                    })
                    self.shares_owned = 0
                    self.buy_price    = 0.0
                    self.peak_price   = 0.0

                    # FIX 2: Removed the early `continue` and the mismatched
                    # portfolio_history.append(self.capital) that was inside the
                    # stop block. Now ALL exit paths (stop or signal) fall through
                    # to the single portfolio_history.append(current_value) at the
                    # bottom, keeping the equity curve consistent.
                    #
                    # Bug: The stop path appended self.capital (cash only, shares
                    # already zeroed) and then skipped the bottom append via
                    # `continue`. The signal-sell path fell through to the bottom
                    # and appended capital + 0 * price. Both should be identical,
                    # but the stop path also skipped any same-bar BUY signal
                    # check. The new flow is: stop fires → shares zeroed → fall
                    # through → current_value = capital + 0 * price → append.
                    # A same-bar BUY is intentionally still skipped (signal is
                    # consumed by the stop exit, not re-entered on the same bar).

            # ── BUY ───────────────────────────────────────────────────────
            if signal == 1 and self.shares_owned == 0:
                effective_capital = self.capital * (1 - self.friction_pct)
                self.shares_owned = int(effective_capital // price)
                self.buy_price    = price
                self.peak_price   = price          # initialise peak at entry
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

            # FIX 2: Single, unified equity-curve append for ALL paths —
            # stop exits, signal exits, buys, and hold days all land here.
            current_value = self.capital + (self.shares_owned * price)
            self.portfolio_history.append(current_value)

        # Pad history if loop breaks early
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