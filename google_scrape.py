import csv
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
nltk.download('vader_lexicon')

def scrape_google_news(company, start_date, end_date, driver):
    """
    Scrape Google News headlines for a given company between start_date and end_date.
    Dates should be in MM/DD/YYYY format.
    Implements pagination by clicking the "Next" button.
    """
    query = company.replace(" ", "+")
    url = f"https://www.google.com/search?q={query}&tbm=nws&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"
    driver.get(url)
    time.sleep(3)  # Wait for the first page to load

    headlines = set()
    # List of selectors to try. You can add more if needed.
    selectors = [
        "a.WlydOe",          # Primary headline link
        "div.JheGif.nDgy9d",  # Alternative headline container
        "div.dbsr"           # Overall container; will extract inner headline text if available
    ]
    
    # Loop through all pages using pagination
    while True:
        # Try each selector on the current page
        for sel in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, sel)
                for elem in elements:
                    # For container elements, try to extract a specific headline child
                    if sel == "div.dbsr":
                        try:
                            title_elem = elem.find_element(By.CSS_SELECTOR, "div.JheGif.nDgy9d")
                            text = title_elem.text.strip()
                        except Exception:
                            text = elem.text.strip()
                    else:
                        text = elem.text.strip()
                    if text:
                        headlines.add(text)
            except Exception as e:
                print(f"Error with selector {sel}: {e}")

        # Try to locate and click the "Next" button
        try:
            next_button = driver.find_element(By.ID, "pnnext")
            next_button.click()
            time.sleep(3)  # Wait for the next page to load
        except Exception:
            # No next button found, exit pagination loop
            break

    return list(headlines)

def analyze_sentiment(headlines):
    """
    Analyze sentiment of each headline using VADER and return a list of dictionaries.
    """
    sid = SentimentIntensityAnalyzer()
    results = []
    for headline in headlines:
        sentiment = sid.polarity_scores(headline)
        compound_score = sentiment['compound']
        results.append({"headline": headline, "sentiment": compound_score})
    return results

def save_to_csv(company, results, filename):
    """
    Save the sentiment results to a CSV file.
    """
    with open(filename, mode="w", newline='', encoding="utf-8") as csv_file:
        fieldnames = ["company", "headline", "sentiment"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({"company": company, "headline": result["headline"], "sentiment": result["sentiment"]})

def main():
    # Configure Chrome options for Selenium
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run headless if you don't need a browser UI
    # Optional: Add a user-agent to mimic a real browser
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36")
    
    # Set up ChromeDriver (update the path to the chromedriver executable)
    service = Service("/Users/wongyule/Documents/Designing Intelligent Agents/FinancialTradingBot/chromedriver-mac-arm64/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Define parameters for scraping
    company = "Apple Inc"
    # Date format must be MM/DD/YYYY
    start_date = "01/01/2022"
    end_date = "01/07/2022"
    
    # Scrape headlines from Google News with pagination
    headlines = scrape_google_news(company, start_date, end_date, driver)
    if not headlines:
        print("No headlines found!")
        driver.quit()
        return
    
    # Analyze sentiment for the extracted headlines
    results = analyze_sentiment(headlines)
    
    # Save the results to a CSV file
    filename = f"{company.replace(' ', '_')}_news_sentiment.csv"
    save_to_csv(company, results, filename)
    print(f"Data saved to {filename}")
    
    driver.quit()

if __name__ == "__main__":
    main()
