import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import numpy as np
import os
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
}

CHART_CONFIG = {
    "figure_size": (16, 14),
    "dpi": 300,
    "style": "seaborn-v0_8-whitegrid",
    "portfolio_color": "#00C853",
    "portfolio_fill": "#00C853",
    "positive_area": "#E8F5E9",
    "negative_area": "#FFEBEE",
    "zero_line_color": "#333333",
    "title_fontsize": 14,
    "subtitle_fontsize": 10,
    "label_fontsize": 10,
    "tick_fontsize": 8,
}


def fetch_benchmark(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch a single benchmark and normalize return to portfolio start date."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"  ⚠️ No data for {ticker}")
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[['Close']].copy()
        data.columns = [ticker]
        return data

    except Exception as e:
        print(f"  ⚠️ Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def load_portfolio_data(csv_path: str) -> pd.DataFrame:
    """Load and clean daily equity data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No daily_equity.csv found at {csv_path}. Run tracker first!")

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop_duplicates(subset='Date', keep='last')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    if len(df) < 2:
        raise ValueError("Need at least 2 days of data.")

    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate multiple return metrics from net worth."""
    start_capital = df['Net_Worth'].iloc[0]
    df['Abs_Return'] = df['Net_Worth'] - start_capital
    df['Pct_Return'] = ((df['Net_Worth'] - start_capital) / start_capital) * 100
    df['Daily_Abs_Change'] = df['Net_Worth'].diff()
    df['Daily_Pct_Change'] = df['Net_Worth'].pct_change() * 100
    df['Running_Max'] = df['Net_Worth'].cummax()
    df['Drawdown_Pct'] = ((df['Net_Worth'] - df['Running_Max']) / df['Running_Max']) * 100
    return df


def generate_combined_chart(csv_path: str = 'config/daily_equity.csv'):
    """
    Generate a combined figure with BOTH charts:
    - LEFT: Portfolio Return Over Time (with drawdown + daily bars)
    - RIGHT: Portfolio vs Global Benchmarks
    """
    print("📊 Loading portfolio data...")
    df_port = load_portfolio_data(csv_path)
    df_port = calculate_returns(df_port)

    dates = df_port.index
    returns = df_port['Pct_Return']
    drawdown = df_port['Drawdown_Pct']
    daily_changes = df_port['Daily_Pct_Change'].fillna(0)
    start_capital = df_port['Net_Worth'].iloc[0]

    start_date = dates[0].strftime('%Y-%m-%d')
    end_date = dates[-1].strftime('%Y-%m-%d')
    days_held = (dates[-1] - dates[0]).days
    final_return = returns.iloc[-1]
    max_return = returns.max()
    max_drawdown = drawdown.min()
    new_high_mask = df_port['Net_Worth'] == df_port['Running_Max']

    cfg = CHART_CONFIG
    plt.style.use(cfg["style"])

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN FIGURE: 1 row, 2 columns
    # ═══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.12)

    # =====================================================================
    # LEFT PANEL: Portfolio Return Over Time
    # =====================================================================
    gs_left = gs[0].subgridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)

    # ── Left Top: Main Return Line ────────────────────────────────────────
    ax1 = fig.add_subplot(gs_left[0])

    ax1.fill_between(dates, 0, returns,
                     where=(returns >= 0),
                     color=cfg["positive_area"], alpha=0.6, interpolate=True)
    ax1.fill_between(dates, 0, returns,
                     where=(returns < 0),
                     color=cfg["negative_area"], alpha=0.6, interpolate=True)

    ax1.plot(dates, returns,
             color=cfg["portfolio_color"],
             linewidth=2.5,
             label='Portfolio Return',
             zorder=5)

    ax1.axhline(0, color=cfg["zero_line_color"], linewidth=1, linestyle='-', alpha=0.5)

    new_high_dates = dates[new_high_mask]
    new_high_values = returns[new_high_mask]
    ax1.scatter(new_high_dates, new_high_values,
                color='#FFD700', s=35, zorder=6, label='New Highs',
                edgecolors='#B8860B', linewidths=0.5)

    ax1.set_title('Portfolio Return Over Time',
                  fontsize=cfg["title_fontsize"], fontweight='bold', pad=10)

    subtitle = (f"{start_date} → {end_date}  |  {days_held} days  |  "
                f"Final: {final_return:+.2f}%  |  Max DD: {max_drawdown:.2f}%")
    ax1.text(0.5, 1.02, subtitle, transform=ax1.transAxes,
             fontsize=cfg["subtitle_fontsize"], ha='center',
             color='#555555', style='italic')

    ax1.set_ylabel('Return (%)', fontsize=cfg["label_fontsize"])
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.tick_params(axis='both', labelsize=cfg["tick_fontsize"])
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Left Middle: Drawdown ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs_left[1], sharex=ax1)

    ax2.fill_between(dates, 0, drawdown,
                     color='#EF5350', alpha=0.4, label='Drawdown')
    ax2.plot(dates, drawdown, color='#C62828', linewidth=1.2, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5, alpha=0.5)

    ax2.set_ylabel('Drawdown (%)', fontsize=cfg["label_fontsize"])
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.tick_params(axis='both', labelsize=cfg["tick_fontsize"])
    ax2.legend(loc='lower left', fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(min(drawdown.min() - 1, -1), 1)

    # ── Left Bottom: Daily Changes ────────────────────────────────────────
    ax3 = fig.add_subplot(gs_left[2], sharex=ax1)

    bar_colors = ['#00C853' if x >= 0 else '#FF1744' for x in daily_changes]
    ax3.bar(dates, daily_changes, color=bar_colors, width=0.8, alpha=0.7, edgecolor='none')
    ax3.axhline(0, color='black', linewidth=0.5, alpha=0.5)

    ax3.set_ylabel('Daily Change (%)', fontsize=cfg["label_fontsize"])
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.set_xlabel('Date', fontsize=cfg["label_fontsize"])
    ax3.tick_params(axis='both', labelsize=cfg["tick_fontsize"])
    ax3.grid(True, alpha=0.3, axis='y')

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=25, ha='right')

    # =====================================================================
    # RIGHT PANEL: Portfolio vs Global Benchmarks
    # =====================================================================
    gs_right = gs[1].subgridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)

    print(f"\n📊 Fetching benchmarks for {start_date} to {end_date}...")

    bench_end = (dates[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    all_benchmarks = pd.DataFrame(index=df_port.index)

    for name, config in BENCHMARKS.items():
        print(f"  📈 {name} ({config['ticker']})...")
        bench = fetch_benchmark(config['ticker'], start_date, bench_end)
        if bench.empty:
            continue

        merged = pd.merge(
            df_port[['Pct_Return']],
            bench,
            left_index=True,
            right_index=True,
            how='outer'
        )
        merged['Pct_Return'] = merged['Pct_Return'].ffill()
        merged.dropna(subset=[config['ticker']], inplace=True)

        start_price = merged[config['ticker']].iloc[0]
        merged[f'{name}_Return'] = ((merged[config['ticker']] - start_price) / start_price) * 100
        all_benchmarks[f'{name}_Return'] = merged[f'{name}_Return'].reindex(df_port.index).ffill()

    # Merge benchmarks into plot df
    df_plot = df_port[['Pct_Return']].copy()
    for col in all_benchmarks.columns:
        df_plot[col] = all_benchmarks[col]

    benchmark_cols = [c for c in df_plot.columns if c != 'Pct_Return']
    df_plot.dropna(subset=benchmark_cols, how='all', inplace=True)

    # ── Right Top: Benchmark Comparison ───────────────────────────────────
    ax4 = fig.add_subplot(gs_right[0])

    ax4.plot(df_plot.index, df_plot['Pct_Return'],
             label='T_Raider Portfolio',
             color='#00ff00',
             linewidth=2.5,
             zorder=10)

    for name, config in BENCHMARKS.items():
        col = f'{name}_Return'
        if col not in df_plot.columns:
            continue
        ax4.plot(df_plot.index, df_plot[col],
                 label=f'{name} (Benchmark)',
                 color=config['color'],
                 linestyle=config['linestyle'],
                 linewidth=1.5,
                 alpha=0.8)

    ax4.axhline(0, color='black', linewidth=1)
    ax4.set_title('T_Raider vs Global Benchmarks',
                  fontsize=cfg["title_fontsize"], fontweight='bold', pad=10)

    ax4.set_ylabel('Return (%)', fontsize=cfg["label_fontsize"])
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax4.tick_params(axis='x', labelbottom=False)
    ax4.tick_params(axis='both', labelsize=cfg["tick_fontsize"])
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Right Middle: Alpha (Portfolio - Benchmark) ───────────────────────
    ax5 = fig.add_subplot(gs_right[1], sharex=ax4)

    alpha_plotted = False
    for name, config in BENCHMARKS.items():
        col = f'{name}_Return'
        if col not in df_plot.columns:
            continue
        alpha = df_plot['Pct_Return'] - df_plot[col]
        ax5.plot(df_plot.index, alpha,
                 label=f'α vs {name}',
                 color=config['color'],
                 linestyle='-',
                 linewidth=1.2,
                 alpha=0.7)
        alpha_plotted = True

    if alpha_plotted:
        ax5.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax5.set_ylabel('Alpha (%)', fontsize=cfg["label_fontsize"])
        ax5.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax5.legend(loc='upper left', fontsize=7)
    else:
        ax5.text(0.5, 0.5, 'No benchmark data', ha='center', va='center',
                 transform=ax5.transAxes, fontsize=10, color='gray')

    ax5.tick_params(axis='x', labelbottom=False)
    ax5.tick_params(axis='both', labelsize=cfg["tick_fontsize"])
    ax5.grid(True, alpha=0.3)

    # ── Right Bottom: Rolling 5-day Alpha ─────────────────────────────────
    ax6 = fig.add_subplot(gs_right[2], sharex=ax4)

    rolling_plotted = False
    for name, config in BENCHMARKS.items():
        col = f'{name}_Return'
        if col not in df_plot.columns:
            continue
        alpha = df_plot['Pct_Return'] - df_plot[col]
        rolling_alpha = alpha.rolling(window=min(5, len(alpha)), min_periods=1).mean()
        ax6.plot(df_plot.index, rolling_alpha,
                 label=f'5d α vs {name}',
                 color=config['color'],
                 linewidth=1.2,
                 alpha=0.7)
        rolling_plotted = True

    if rolling_plotted:
        ax6.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax6.set_ylabel('5d Roll α (%)', fontsize=cfg["label_fontsize"])
        ax6.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax6.legend(loc='upper left', fontsize=7)
    else:
        ax6.text(0.5, 0.5, 'No benchmark data', ha='center', va='center',
                 transform=ax6.transAxes, fontsize=10, color='gray')

    ax6.set_xlabel('Date', fontsize=cfg["label_fontsize"])
    ax6.tick_params(axis='both', labelsize=cfg["tick_fontsize"])
    ax6.grid(True, alpha=0.3, axis='y')

    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax6.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=25, ha='right')

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════════════
    plt.tight_layout()
    chart_path = 'config/performance_dashboard.png'
    plt.savefig(chart_path, dpi=cfg["dpi"], facecolor='white', bbox_inches='tight')
    print(f"\n✅ Combined dashboard saved to {chart_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("T_RAIDER PERFORMANCE DASHBOARD SUMMARY")
    print("=" * 60)
    print(f"{'Period':<28} {start_date} to {end_date}")
    print(f"{'Trading Days':<28} {len(df_port)}")
    print(f"{'Calendar Days':<28} {days_held}")
    print("-" * 60)
    print(f"{'Start Capital':<28} ₹{start_capital:,.2f}")
    print(f"{'End Capital':<28} ₹{df_port['Net_Worth'].iloc[-1]:,.2f}")
    print(f"{'Total Return':<28} {final_return:+.2f}%")
    print(f"{'Absolute P&L':<28} ₹{df_port['Abs_Return'].iloc[-1]:+,.2f}")
    print("-" * 60)
    print(f"{'Max Return (peak)':<28} {max_return:+.2f}%")
    print(f"{'Max Drawdown':<28} {max_drawdown:.2f}%")
    print(f"{'Current Drawdown':<28} {drawdown.iloc[-1]:.2f}%")
    print(f"{'New Highs Reached':<28} {new_high_mask.sum()}")
    print(f"{'Best Day':<28} +{daily_changes.max():.2f}%")
    print(f"{'Worst Day':<28} {daily_changes.min():.2f}%")
    print(f"{'Avg Daily Change':<28} {daily_changes.mean():+.3f}%")
    print(f"{'Volatility (σ daily)':<28} {daily_changes.std():.3f}%")
    print("-" * 60)

    for name in BENCHMARKS.keys():
        col = f'{name}_Return'
        if col in df_plot.columns:
            final_bench = df_plot[col].iloc[-1]
            alpha = final_return - final_bench
            print(f"{'vs ' + name:<28} {final_bench:>+9.2f}%  (α: {alpha:+.2f}%)")

    if days_held > 30:
        ann_return = ((df_port['Net_Worth'].iloc[-1] / start_capital) ** (365.25 / days_held) - 1) * 100
        ann_vol = daily_changes.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        print("-" * 60)
        print(f"{'Annualized Return':<28} {ann_return:+.2f}%")
        print(f"{'Annualized Volatility':<28} {ann_vol:.2f}%")
        print(f"{'Sharpe (approx)':<28} {sharpe:.2f}")

    print("=" * 60)

    plt.close()
    return df_port


if __name__ == "__main__":
    generate_combined_chart()