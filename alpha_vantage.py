import os
import requests
import pandas as pd

API_KEY = "UJEWN0C26N1AWRMV"

BASE_URL = "https://www.alphavantage.co/query"

SAVE_FOLDER = "stock_data"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def get_stock_data(symbol, start_date="2022-01-01", end_date="2024-12-31", output_size="full", save_csv=True):
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": output_size,
        "apikey": API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if "Time Series (Daily)" in data:
        # Create DataFrame
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        
        # Convert index to datetime and columns to float
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        
        # IMPORTANT: Sort the index to ascending (oldest -> newest)
        df.sort_index(inplace=True)
        
        # Filter by date
        df_filtered = df.loc[start_date:end_date]
        
        if save_csv:
            csv_filename = os.path.join(SAVE_FOLDER, f"{symbol}_stock_data_{start_date}_to_{end_date}.csv")
            df_filtered.to_csv(csv_filename)
            print(f"[INFO] Data for {symbol} saved to {csv_filename} ({len(df_filtered)} records)")
        
        return df_filtered
    else:
        print(f"[ERROR] Could not fetch data for {symbol}. Response: {data}")
        return None

# Dictionary mapping company names to tickers
company_tickers = {
    "American Express Company": "AXP",
    "Visa Inc.": "V",
    "Apple Inc.": "AAPL",
    "Chevron Corporation": "CVX",
    "Exxon Mobil Corporation": "XOM",
    "McDonalds Corporation": "MCD",
    "Pfizer Inc": "PFE",
    "Merck & Co. Inc.": "MRK",
    "Walgreens Boots Alliance Inc.": "WBA",
    "Nike Inc.": "NKE",
    "UnitedHealth Group Inc.": "UNH",
}

if __name__ == "__main__":
    start_date = "2022-01-01"
    end_date = "2024-12-31"
    
    for company_name, ticker in company_tickers.items():
        print(f"[INFO] Fetching data for: {company_name} ({ticker})")
        df_stock = get_stock_data(
            ticker, 
            start_date=start_date, 
            end_date=end_date, 
            output_size="full", 
            save_csv=True
        )
        
        if df_stock is not None:
            print(df_stock.head(), "\n")
