import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mtick

def generate_performance_chart():
    csv_path = 'config/daily_equity.csv'
    
    if not os.path.exists(csv_path):
        print("No daily_equity.csv found. Run the tracker first!")
        return

    # 1. Load Portfolio Data
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

    # 2. Fetch Nifty 50 Data for the exact same time period
    start_date = df_port.index[0].strftime('%Y-%m-%d')
    end_date = (df_port.index[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d') # Add 1 day to include today
    
    nifty = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
    
    if nifty.empty:
        print("Failed to fetch Nifty data for comparison.")
        return

    # Clean multi-index if necessary
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
        
    nifty = nifty[['Close']].rename(columns={'Close': 'Nifty_Close'})
    
    # 3. Merge Portfolio and Nifty data
    # We use forward-fill so weekends carry over the Friday portfolio balance
    df_merged = pd.merge(df_port[['Portfolio_Return']], nifty, left_index=True, right_index=True, how='outer')
    df_merged['Portfolio_Return'] = df_merged['Portfolio_Return'].ffill()
    df_merged.dropna(subset=['Nifty_Close'], inplace=True) # Only plot on trading days
    
    # Calculate Nifty % Return matching our start date
    start_nifty = df_merged['Nifty_Close'].iloc[0]
    df_merged['Nifty_Return'] = ((df_merged['Nifty_Close'] - start_nifty) / start_nifty) * 100

    # 4. Plot the Graph
    plt.figure(figsize=(10, 5))
    plt.plot(df_merged.index, df_merged['Portfolio_Return'], label='T_Raider Portfolio', color='#00ff00', linewidth=2.5)
    plt.plot(df_merged.index, df_merged['Nifty_Return'], label='Nifty 50 (Benchmark)', color='#555555', linestyle='--', linewidth=2)
    
    # Formatting
    plt.title('T_Raider vs Nifty 50 Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper left')
    
    # Save the image
    plt.tight_layout()
    chart_path = 'config/performance_chart.png'
    plt.savefig(chart_path, dpi=300, facecolor='white')
    print(f"✅ Success! Chart saved to {chart_path}")

if __name__ == "__main__":
    generate_performance_chart()