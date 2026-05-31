import requests
import pandas as pd
import time
import PyPDF2
from io import BytesIO

# --- CONFIGURATION ---
high_impact_keywords = [
    "Resignation", 
    "Acquisition", 
    "Pledge", 
    "Capex", 
    "Capacity", 
    "Credit Rating", 
    "Default", 
    "Merger", 
    "Auditor",
    "Expansion"
]

# --- CORE FUNCTIONS ---
def fetch_daily_nse_filings(session):
    """Fetches the current day's corporate announcements using the shared session."""
    try:
        print("Fetching routing cookies from NSE homepage...")
        session.get("https://www.nseindia.com", timeout=10)
        
        time.sleep(2)
        
        print("Fetching today's corporate announcements...")
        api_url = "https://www.nseindia.com/api/corporate-announcements?index=equities"
        response = session.get(api_url, timeout=15)
        
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            print(f"Failed to fetch. Status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"An error occurred during fetch: {e}")
        return None
    
def filter_high_impact_filings(df, keywords):
    """Filters the DataFrame for specific high-impact keywords using Regex."""
    if df is None or df.empty:
        return df
        
    print(f"Filtering {len(df)} total filings for high-impact events...")
        
    pattern = '|'.join(keywords)
    df['attchmntText'] = df['attchmntText'].fillna('')
    df['desc'] = df['desc'].fillna('')
    
    filtered_df = df[
        df['attchmntText'].str.contains(pattern, case=False, na=False) |
        df['desc'].str.contains(pattern, case=False, na=False)
    ]
    
    return filtered_df

def extract_text_from_pdf(session, pdf_url):
    """Downloads a PDF using the shared session and extracts its text."""
    if not pdf_url.startswith("http"):
        pdf_url = "https://www.nseindia.com" + pdf_url

    print(f"\nDownloading PDF: {pdf_url}")
    
    try:
        response = session.get(pdf_url, timeout=20)
        
        if response.status_code == 200:
            pdf_file = BytesIO(response.content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            extracted_text = ""
            # Limit to the first 5 pages to keep LLM token limits in check
            num_pages = min(len(reader.pages), 5) 
            for page_num in range(num_pages):
                extracted_text += reader.pages[page_num].extract_text() + "\n"
                
            return extracted_text.strip()
        else:
            print(f"Failed to download PDF. Status: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None

# --- DAILY EXECUTION ---
if __name__ == "__main__":
    print(f"--- Starting NSE Daily Pipeline: {time.strftime('%Y-%m-%d')} ---")
    
    # 1. Initialize the Master Session globally
    master_session = requests.Session()
    master_session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    })
    
    # 2. Fetch live data (passing the master_session)
    daily_df = fetch_daily_nse_filings(master_session)
    
    # 3. Filter the DataFrame
    if daily_df is not None and not daily_df.empty:
        filtered_df = filter_high_impact_filings(daily_df, high_impact_keywords)
        
        print(f"\nHigh-impact filings found: {len(filtered_df)}")
        
        if not filtered_df.empty:
            # 4. Iterate over the matches and extract the PDF text
            for index, row in filtered_df.iterrows():
                symbol = row['symbol']
                pdf_link = row['attchmntFile']
                
                pdf_text = extract_text_from_pdf(master_session, pdf_link)
                
                if pdf_text:
                    print(f"Success! Extracted {len(pdf_text)} characters for {symbol}.")
                    # Print a tiny preview to verify text extraction worked
                    print(f"Preview: {pdf_text[:150]}...")
                else:
                    print(f"Failed to extract readable text for {symbol}. It might be a scanned image.")
        else:
            print("No actionable, high-impact filings found today.")
    else:
        print("Pipeline aborted: No data retrieved from the NSE.")