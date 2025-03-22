import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ssl
import nltk

# SSL Fix for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('vader_lexicon')

def click_next_page(driver):
    """
    Attempt to click the "Next" button (id='pnnext') on Google News search results.
    Returns True if successful, False otherwise.
    """
    try:
        # Wait for the next button to be clickable
        next_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "pnnext"))
        )
        next_button.click()
        time.sleep(2)  # Allow time for the next page to load
        return True
    except Exception as e:
        print("Next page not found or not clickable:", e)
        return False

def scrape_google_news(company, driver):
    """
    Scrape Google News headlines for a given company using a date range.
    Uses Google search with tbm=nws and tbs parameters to filter news between 2022 and 2024.
    Iterates through pages using the "Next" button.
    """
    query = company.replace(" ", "+")
    # URL with date range filtering: 01/01/2022 to 12/31/2024
    url = (
        f"https://www.google.com/search?q={query}"
        f"&tbm=nws&tbs=cdr:1,cd_min:{"01/01/2022"},cd_max:{"12/31/2024"},sbd:1"
    )
    driver.get(url)
    time.sleep(2)  # Wait for the page to load
    
    headlines = set()
    # Selectors known to capture headlines on Google News search results
    selectors = [
        "a.WlydOe",          # Primary headline link
        "div.JheGif.nDgy9d",  # Headline container
        "div.dbsr"           # Overall container; extract headline text inside it
    ]
    
    page_number = 1
    while True:
        print(f"Scraping page {page_number} for {company}...")
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.dbsr"))
            )
        except Exception as e:
            print(f"Error waiting for articles for {company} on page {page_number}: {e}")
        
        time.sleep(2)  # Extra wait to ensure content is loaded
        
        for sel in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, sel)
                for elem in elements:
                    # For overall container, try to extract child headline element
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
                print(f"Error with selector {sel} for {company} on page {page_number}: {e}")
        
        # Attempt to click the next page; if not found, exit loop
        if not click_next_page(driver):
            break
        
        page_number += 1
    
    return list(headlines)

def analyze_sentiment(headlines):
    """
    Analyze sentiment of each headline using VADER.
    """
    sid = SentimentIntensityAnalyzer()
    results = []
    for headline in headlines:
        sentiment = sid.polarity_scores(headline)
        compound_score = sentiment['compound']
        results.append({"headline": headline, "sentiment": compound_score})
    return results

def save_to_csv(company, results):
    """
    Save sentiment results to a CSV file.
    """
    filename = f"{company.replace(' ', '_')}_news_sentiment_1.csv"
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
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Set up ChromeDriver (update the path as needed)
    service = Service("/Users/wongyule/Documents/Designing Intelligent Agents/FinancialTradingBot/chromedriver-mac-arm64/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # List of companies from the Dow Jones portfolio
    companies = [
        "Chevron"
    ]

    for company in companies:
        print(f"Scraping news for {company}...")
        headlines = scrape_google_news(company, driver)
        if not headlines:
            print(f"No headlines found for {company}")
            continue

        # Perform sentiment analysis
        results = analyze_sentiment(headlines)
        # Save results to CSV file
        save_to_csv(company, results)

    driver.quit()

if __name__ == "__main__":
    main()