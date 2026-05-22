import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mtick
import sys

# ── Path fix for imports ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from macro_filter import MARKET_CONFIGS  # Reuse market definitions if available


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — Add/remove benchmarks here
# ═════════════════════════════════════════════════════════════════════════════

BENCHMARKS = {
    "Nifty 50": {
        "ticker": "^NSEI",
        "color": "#555555",
        "linestyle": "--",
    },
    "S&P 500": {
        "ticker": "^GSPC",
        "color": "#1f77b4",  # Blue
        "linestyle": "-.",
    },
    "NASDAQ": {
        "ticker": "^IXIC",
        "color": "#ff7f0e",  # Orange
        "linestyle": "-.",
    },
    "Dow Jones": {
        "ticker": "^DJI",
        "color": "#9467bd",  # Purple
        "linestyle": "-.",
    },
    # Add more as needed:
    # "Russell 2000": {"ticker": "^RUT", "color": "#2ca02c", "linestyle": ":"},
    # "FTSE 100": {"ticker": "^FTSE", "color": "#d62728", "linestyle": ":"},
    # "Nikkei 225": {"ticker": "^N225", "color": "#8c564b", "linestyle": ":"},
}


def fetch_benchmark(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch a single benchmark and normalize return to portfolio start date."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"  ⚠️ No data for {ticker}")
            return pd.DataFrame()

        # Normalize column names
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[['Close']].copy()
        data.columns = [ticker]
        return data

    except Exception as e:
        print(f"  ⚠️ Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def generate_performance_chart():
    csv_path = 'config/daily_equity.csv'

    if not os.path.exists(csv_path):
        print("No daily_equity.csv found. Run the tracker first!")
        return

    # ── 1. Load Portfolio Data ─────────────────────────────────────────────
    df_port = pd.read_csv(csv_path)
    df_port['Date'] = pd.to_datetime(df_port['Date'])

    # Drop duplicate dates (in case you ran the bot twice in one day)
    df_port = df_port.drop_duplicates(subset='Date', keep='last')
    df_port.set_index('Date', inplace=True)

    if len(df_port) < 2:
        print("Need at least 2 days of data to draw a chart.")
        return

    # Calculate Portfolio % Return
    start_capital = df_port['Net_Worth'].iloc[0]
    df_port['Portfolio_Return'] = ((df_port['Net_Worth'] - start_capital) / start_capital) * 100

    # ── 2. Fetch All Benchmarks ────────────────────────────────────────────
    start_date = df_port.index[0].strftime('%Y-%m-%d')
    end_date = (df_port.index[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"📊 Fetching benchmarks for {start_date} to {end_date}...")

    all_benchmarks = pd.DataFrame(index=df_port.index)

    for name, config in BENCHMARKS.items():
        print(f"  📈 {name} ({config['ticker']})...")
        bench = fetch_benchmark(config['ticker'], start_date, end_date)
        if bench.empty:
            continue

        # Calculate % return from portfolio start date
        merged = pd.merge(
            df_port[['Portfolio_Return']],
            bench,
            left_index=True,
            right_index=True,
            how='outer'
        )
        merged['Portfolio_Return'] = merged['Portfolio_Return'].ffill()
        merged.dropna(subset=[config['ticker']], inplace=True)

        start_price = merged[config['ticker']].iloc[0]
        merged[f'{name}_Return'] = ((merged[config['ticker']] - start_price) / start_price) * 100

        # Reindex to portfolio dates and forward-fill
        all_benchmarks[f'{name}_Return'] = merged[f'{name}_Return'].reindex(df_port.index).ffill()

    # ── 3. Merge Everything ────────────────────────────────────────────────
    df_plot = df_port[['Portfolio_Return']].copy()
    for col in all_benchmarks.columns:
        df_plot[col] = all_benchmarks[col]

    # Drop rows where we have no benchmark data at all
    benchmark_cols = [c for c in df_plot.columns if c != 'Portfolio_Return']
    df_plot.dropna(subset=benchmark_cols, how='all', inplace=True)

    # ── 4. Plot ────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 6))

    # Portfolio (always plotted)
    plt.plot(
        df_plot.index,
        df_plot['Portfolio_Return'],
        label='T_Raider Portfolio',
        color='#00ff00',
        linewidth=2.5,
        zorder=10  # Draw on top
    )

    # Benchmarks
    for name, config in BENCHMARKS.items():
        col = f'{name}_Return'
        if col not in df_plot.columns:
            continue
        plt.plot(
            df_plot.index,
            df_plot[col],
            label=f'{name} (Benchmark)',
            color=config['color'],
            linestyle=config['linestyle'],
            linewidth=1.5,
            alpha=0.8
        )

    # Formatting
    plt.title('T_Raider vs Global Benchmarks', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper left', fontsize=9)

    # Save
    plt.tight_layout()
    chart_path = 'config/performance_chart.png'
    plt.savefig(chart_path, dpi=300, facecolor='white')
    print(f"\n✅ Success! Chart saved to {chart_path}")

    # Print summary stats
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<25} {'Return':>10}")
    print("-" * 50)

    final_portfolio = df_plot['Portfolio_Return'].iloc[-1]
    print(f"{'T_Raider Portfolio':<25} {final_portfolio:>+9.2f}%")

    for name in BENCHMARKS.keys():
        col = f'{name}_Return'
        if col in df_plot.columns:
            final = df_plot[col].iloc[-1]
            alpha = final - final_portfolio
            print(f"{name + ' (Benchmark)':<25} {final:>+9.2f}%  (α: {alpha:+.2f}%)")

    print("=" * 50)


if __name__ == "__main__":
    generate_performance_chart()