import csv
import time
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ssl
import nltk

# SSL fix for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('vader_lexicon')

def extract_article_data(elem):
    """
    Attempt to extract headline text from a list of possible selectors, then
    fall back to using the container element's text if all else fails.
    Finally, attempt to parse a <time> element if present for the article date.
    """
    selectors = [
        "a.WlydOe",           # Primary headline link
        "div.JheGif.nDgy9d", # Headline container
        "div.dbsr"          # Typically the entire container, so it's used last as a fallback
    ]
    
    headline = ""
    for selector in selectors:
        try:
            sub_elem = elem.find_element(By.CSS_SELECTOR, selector)
            text = sub_elem.text.strip()
            if text:
                headline = text
                break
        except Exception:
            pass

    # If no headline was found using the above selectors, fallback to the container's text.
    if not headline:
        headline = elem.text.strip()

    # Attempt to extract date from <time> element.
    article_date = None
    try:
        time_elem = elem.find_element(By.TAG_NAME, "time")
        date_str = time_elem.get_attribute("datetime")
        if date_str:
            # Remove trailing 'Z' if present
            if date_str.endswith("Z"):
                date_str = date_str[:-1]
            article_date = datetime.fromisoformat(date_str)
    except Exception:
        pass

    return {"headline": headline, "date": article_date}

def scrape_google_news_range(company, cd_min, cd_max, driver, max_pages=30):

    query = company.replace(" ", "+")
    url = f"https://www.google.com/search?q={query}&tbm=nws&tbs=cdr:1,cd_min:{cd_min},cd_max:{cd_max},sbd:1"
    driver.get(url)
    
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.dbsr"))
        )
    except Exception as e:
        print(f"Error waiting for articles for {company} between {cd_min} and {cd_max}: {e}")
    time.sleep(2)
    
    articles = []
    page_number = 1
    while page_number <= max_pages:
        print(f"Scraping page {page_number} for {company} (date range {cd_min} to {cd_max})...")
        article_elements = driver.find_elements(By.CSS_SELECTOR, "div.dbsr")
        for elem in article_elements:
            data = extract_article_data(elem)
            if data["headline"]:
                articles.append(data)

        if page_number == max_pages:
            # We reached the max page limit for this date range
            print(f"Reached max page limit ({max_pages}) for {company} in this date range.")
            break

        # Try to click the "Next" button
        try:
            next_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "pnnext"))
            )
            next_button.click()
            time.sleep(2)
            page_number += 1
        except Exception as e:
            print(f"No next page for {company} in this date range: {e}")
            break

    # If the loop ended before hitting page_number == max_pages,
    # it means we didn't reach the final (30th) page in the date range.
    reached_max_page = (page_number == max_pages)
    return articles, reached_max_page

def analyze_sentiment_for_articles(articles):
    sid = SentimentIntensityAnalyzer()
    for article in articles:
        sentiment = sid.polarity_scores(article["headline"])
        article["sentiment"] = sentiment["compound"]
    return articles

def save_articles_to_csv(company, articles):
    filename = f"{company.replace(' ', '_')}_news_sentiment.csv"
    with open(filename, mode="w", newline='', encoding="utf-8") as csv_file:
        fieldnames = ["company", "headline", "date", "sentiment"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for article in articles:
            date_str = article["date"].strftime("%m/%d/%Y") if article["date"] else ""
            writer.writerow({
                "company": company,
                "headline": article["headline"],
                "date": date_str,
                "sentiment": article.get("sentiment", "")
            })
    print(f"Data saved to {filename}")

def main():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) " \
                                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.5304.87 Safari/537.36")
    service = Service("/Users/wongyule/Documents/Designing Intelligent Agents/FinancialTradingBot/chromedriver-mac-arm64/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    companies = [
        "3M Corporation", "American Express Company", "Travellers Companies Inc.", "Visa Inc.",
        "JP Morgan Chase & Co.", "Goldman Sachs Group Inc.", "Apple Inc.", "Microsoft Corporation",
        "Intel Corporation", "IBM Corporation", "Cisco Systems Inc.", "Boeing Corporation",
        "Raytheon Technologies Corporation", "Caterpillar Inc.", "Chevron Corporation",
        "Exxon Mobil Corporation", "McDonalds Corporation", "Coca-Cola Corporation",
        "Johnson & Johnson Corporation", "Pfizer Inc", "Merck & Co. Inc.", "DuPont de Nemours Inc",
        "Walgreens Boots Alliance Inc.", "Walmart Inc.", "Home Depot Inc.", "Nike Inc.",
        "UnitedHealth Group Inc.", "Proctor & Gamble Corporation", "Verizon Communications Inc.",
        "Walt Disney Company"
    ]
    
    # Define the fixed overall date range: 01/01/2022 to 12/31/2024
    cd_min_date = datetime.strptime("10/31/2024", "%m/%d/%Y")
    cd_max_date = datetime.strptime("12/31/2024", "%m/%d/%Y")
    cd_min_str = cd_min_date.strftime("%m/%d/%Y")
    
    for company in companies:
        print(f"\nProcessing {company}...")
        all_articles = []
        current_cd_max = cd_max_date

        while current_cd_max > cd_min_date:
            cd_max_str = current_cd_max.strftime("%m/%d/%Y")
            print(f"Scraping articles for {company} with date range {cd_min_str} to {cd_max_str}...")
            
            # --- CHANGED HERE: We unpack reached_max_page as well ---
            articles, reached_max_page = scrape_google_news_range(company, cd_min_str, cd_max_str, driver, max_pages=30)

            # Always extend all_articles with whatever was scraped
            all_articles.extend(articles)

            # Check if we reached the max page. If not, skip to next company.
            if not reached_max_page:
                print(f"Max page reached is not 30 for {company}. Jumping to next company...")
                break

            if not articles:
                print(f"No articles found for {company} in this date range. Decrementing date by one day and trying again.")
                current_cd_max = current_cd_max - timedelta(days=1)
                continue

            # Filter for articles with valid dates
            dated_articles = [a for a in articles if a["date"] is not None]
            if not dated_articles:
                print(f"No dated articles found for {company} in this range. Decrementing date by one day and trying again.")
                current_cd_max = current_cd_max - timedelta(days=1)
                continue

            oldest_article = min(dated_articles, key=lambda a: a["date"])
            print(f"Oldest article date found: {oldest_article['date'].strftime('%m/%d/%Y')}")
            # Set new cd_max to one day before the oldest article date
            current_cd_max = oldest_article["date"] - timedelta(days=1)
            if current_cd_max <= cd_min_date:
                break

        # Remove duplicates and sort articles by date (oldest first)
        unique_articles = {(a["headline"], a["date"]): a for a in all_articles}.values()
        sorted_articles = sorted(
            unique_articles,
            key=lambda a: a["date"] if a["date"] is not None else datetime.max
        )
        sorted_articles = analyze_sentiment_for_articles(sorted_articles)
        save_articles_to_csv(company, sorted_articles)
    
    driver.quit()

if __name__ == "__main__":
    main()
