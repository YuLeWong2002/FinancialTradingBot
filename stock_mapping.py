import os
import pandas as pd

# Mapping of full company names to tickers
company_to_ticker = {
    "3M Corporation": "MMM",
    "American Express Company": "AXP",
    "Travellers Companies Inc.": "TRV",
    "Visa Inc.": "V",
    "JP_Morgan_Chase": "JPM",
    "Goldman Sachs Group Inc.": "GS",
    "Apple Inc.": "AAPL",
    "Microsoft Corporation": "MSFT",
    "Intel Corporation": "INTC",
    "IBM Corporation": "IBM",
    "Cisco_Systems": "CSCO",
    "Boeing": "BA",
    "Raytheon": "RTX",
    "Caterpillar": "CAT",
    "Chevron": "CVX",
    "Exxon Mobil": "XOM",
    "McDonalds": "MCD",
    "Coca-cola": "KO",
    "Johnson": "JNJ",
    "Pfizer": "PFE",
    "Merck": "MRK",
    "Dupont_de_Nemours": "DD",
    "Walgreens_Boots_Alliance": "WBA",
    "Walmart": "WMT",
    "Home_Depot": "HD",
    "Nike": "NKE",
    "UnitedHealth_Group": "UNH",
    "Proctor_Gamble": "PG",
    "Verizon_Communications": "VZ",
    "Walt_Disney": "DIS"
}

# Inverted mapping for quick lookup by partial name in filename
name_lookup = {name.replace(" ", "_"): ticker for name, ticker in company_to_ticker.items()}

# Path to your sentiment data
news_folder = "news_sentiment"

# Loop through all CSV files in the folder
for filename in os.listdir(news_folder):
    if not filename.endswith(".csv"):
        continue

    full_path = os.path.join(news_folder, filename)

    # Try to find a matching company name in the filename
    matched_ticker = None
    for partial_name, ticker in name_lookup.items():
        if partial_name in filename:
            matched_ticker = ticker
            break

    if matched_ticker is None:
        print(f"[WARNING] No match found for {filename}, skipping.")
        continue

    # Load CSV and standardize company column
    df = pd.read_csv(full_path)
    if "company" in df.columns:
        df["company"] = matched_ticker
        df.to_csv(full_path, index=False)
        print(f"[INFO] Standardized company column in {filename} → {matched_ticker}")
    
    # Optionally rename the file to start with the ticker
    new_filename = f"{matched_ticker}_news_sentiment.csv"
    new_path = os.path.join(news_folder, new_filename)
    os.rename(full_path, new_path)
    print(f"[INFO] Renamed file: {filename} → {new_filename}")
