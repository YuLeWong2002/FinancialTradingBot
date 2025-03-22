import csv
import time
import ssl
from datetime import datetime, timedelta

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('vader_lexicon')

def reduce_date_range(start_str, end_str, days_to_reduce=60):
    start_dt = datetime.strptime(start_str, "%m/%d/%Y")
    end_dt = datetime.strptime(end_str, "%m/%d/%Y")
    new_end_dt = end_dt - timedelta(days=days_to_reduce)
    
    if new_end_dt < start_dt:
        print(f"[INFO] Cannot reduce date any further.")
        return start_str, end_str  # Return unchanged or handle as needed
    
    return start_str, new_end_dt.strftime("%m/%d/%Y")

def build_google_news_url(query, start_date_str, end_date_str):
    return (
        f"https://www.google.com/search?q={query}"
        f"&tbm=nws"
        f"&tbs=cdr:1,cd_min:{start_date_str},cd_max:{end_date_str},sbd:1"
    )

def click_next_page(driver):
    try:
        next_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "pnnext"))
        )
        next_button.click()
        time.sleep(2)
        return True
    except Exception as e:
        print("Next page not found or not clickable:", e)
        return False

def scrape_google_news(company, driver, start_date_str="01/01/2022", end_date_str="12/31/2024"):
    """
    Scrape Google News headlines for a given company, automatically reducing date range
    by 60 days once we reach page 30, and continuing until we cannot reduce further.
    """
    query = company.replace(" ", "+")
    
    # We'll keep a container for all unique headlines
    all_headlines = set()
    
    while True:
        # Build URL for current date range
        url = build_google_news_url(query, start_date_str, end_date_str)
        driver.get(url)
        time.sleep(2)  # Wait for the page to load

        page_number = 1
        
        while True:
            print(f"Scraping page {page_number} for {company} (Range: {start_date_str} to {end_date_str})...")
            
            # Wait for articles
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.dbsr"))
                )
            except Exception as e:
                print(f"Error waiting for articles on page {page_number} for {company}: {e}")
            
            time.sleep(2)
            
            # Collect headlines
            selectors = ["a.WlydOe", "div.JheGif.nDgy9d", "div.dbsr"]
            for sel in selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, sel)
                    for elem in elements:
                        if sel == "div.dbsr":
                            try:
                                # Attempt to extract sub-element for the headline
                                title_elem = elem.find_element(By.CSS_SELECTOR, "div.JheGif.nDgy9d")
                                text = title_elem.text.strip()
                            except Exception:
                                text = elem.text.strip()
                        else:
                            text = elem.text.strip()
                        
                        if text:
                            all_headlines.add(text)
                except Exception as e:
                    print(f"Error with selector {sel} on page {page_number} for {company}: {e}")
            
            # If we reached page 30, break & reduce date range
            if page_number == 30:
                print(f"[INFO] Reached page 30 for {company}. Reducing end date by 60 days...")
                # Reduce date range
                old_start, old_end = start_date_str, end_date_str
                new_start, new_end = reduce_date_range(old_start, old_end, days_to_reduce=60)
                
                # If the new end is the same as old end, we can't reduce further => exit
                if new_end == old_end:
                    print("[INFO] Cannot reduce any further. Stopping.")
                    return list(all_headlines)
                
                # Update date range
                start_date_str, end_date_str = new_start, new_end
                # Break inner loop to start from new date range
                break
            
            # Try to click next page
            if not click_next_page(driver):
                # No more pages => break out & possibly reduce date range again or end
                return list(all_headlines)
            
            page_number += 1

def analyze_sentiment(headlines):
    sid = SentimentIntensityAnalyzer()
    results = []
    for headline in headlines:
        sentiment = sid.polarity_scores(headline)
        compound_score = sentiment['compound']
        results.append({"headline": headline, "sentiment": compound_score})
    return results

def save_to_csv(company, results):
    filename = f"{company.replace(' ', '_')}_news_sentiment_rolling_dates.csv"
    with open(filename, mode="w", newline='', encoding="utf-8") as csv_file:
        fieldnames = ["company", "headline", "sentiment"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "company": company,
                "headline": result["headline"],
                "sentiment": result["sentiment"]
            })
    print(f"Data saved to {filename}")

def main():
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    service = Service("/Users/wongyule/Documents/Designing Intelligent Agents/FinancialTradingBot/chromedriver-mac-arm64/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)

    companies = [
        "Walmart",
        "Home Depot",
        "Nike",
        "UnitedHealth Group",
        "Procter Gamble",
        "Verizon Communications",
        "Walt Disney"
    ]

    for company in companies:
        print(f"Scraping news for {company}...")
        headlines = scrape_google_news(company, driver)
        if not headlines:
            print(f"No headlines found for {company}")
            continue
        
        results = analyze_sentiment(headlines)
        save_to_csv(company, results)

    driver.quit()

if __name__ == "__main__":
    main()
