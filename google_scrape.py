import csv
import time
import random
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

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('vader_lexicon')

def click_next_page(driver):
    """
    Attempts to find and click the 'Next' button on Google News search results.
    Returns True if found/clicked (i.e., can move to next page), False otherwise.
    """
    try:
        next_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "pnnext"))
        )
        time.sleep(random.uniform(1, 2))
        next_button.click()
        time.sleep(random.uniform(2, 4))
        return True
    except Exception:
        return False

def extract_article_data(elem):
    """
    Extract headline & date from a Google News snippet, if present.
    """
    try:
        title_elem = elem.find_element(By.CSS_SELECTOR, "div.JheGif.nDgy9d")
        headline = title_elem.text.strip()
    except Exception:
        headline = elem.text.strip()

    article_date = None
    # Attempt <time datetime="...">
    try:
        time_elem = elem.find_element(By.TAG_NAME, "time")
        date_str = time_elem.get_attribute("datetime")
        if date_str:
            # Example: "2023-11-20T10:00:00Z"
            article_date = datetime.fromisoformat(date_str.rstrip("Z"))
    except Exception:
        # Fallback: snippet container
        try:
            alt_date_elem = elem.find_element(By.CSS_SELECTOR, "div.OSrXXb.rbYSKb.LfVVr span")
            date_text = alt_date_elem.text.strip()  # e.g. "20 Nov 2023"
            article_date = datetime.strptime(date_text, "%d %b %Y")
        except Exception:
            pass

    return {
        "headline": headline,
        "date": article_date
    }

def scrape_google_news_range(company, cd_min, cd_max, driver, max_pages=30):
    """
    Scrape Google News articles for 'company' between cd_min and cd_max (MM/DD/YYYY).
    Up to 'max_pages' pages. If we fill page 30, returns (all_articles, True). Otherwise, (all_articles, False).
    """
    query = company.replace(" ", "+")
    url = (
        f"https://www.google.com/search?q={query}"
        f"&tbm=nws&tbs=cdr:1,cd_min:{cd_min},cd_max:{cd_max},sbd:1"
    )
    print(f"Visiting: {url}")

    # Known selectors that grab headlines
    selectors = ["a.WlydOe", "div.JheGif.nDgy9d", "div.dbsr"]

    # Navigate & wait
    try:
        driver.get(url)
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.dbsr"))
        )
    except Exception as e:
        print(f"⚠️ Possibly blocked or no results for {company}: {e}")
        return [], False

    time.sleep(random.uniform(2, 4))

    all_articles = []
    page_number = 1

    while page_number <= max_pages:
        print(f"📄 Page {page_number} for {company} (range {cd_min} to {cd_max})...")
        # Scroll
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(3, 5))

        page_articles = []

        # Extract headlines using multiple selectors
        for sel in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, sel)
                for elem in elements:
                    if sel == "div.dbsr":
                        # Child element for the container
                        try:
                            title_elem = elem.find_element(By.CSS_SELECTOR, "div.JheGif.nDgy9d")
                            text = title_elem.text.strip()
                        except Exception:
                            text = elem.text.strip()
                    else:
                        text = elem.text.strip()

                    if text:
                        page_articles.append({
                            "headline": text,
                            "date": None  # We'll store date in the fallback approach if needed
                        })
            except Exception as e:
                print(f"Error with selector {sel} on page {page_number}: {e}")

        # Or if you want the date from each snippet, use your 'extract_article_data' approach
        # but that can be slower because it's repeated. Example:
        # page_articles2 = []
        # snippet_elems = driver.find_elements(By.CSS_SELECTOR, "div.dbsr")
        # for elem in snippet_elems:
        #    data = extract_article_data(elem)
        #    if data["headline"]:
        #        page_articles2.append(data)
        # ...
        # We'll keep it simpler for now.

        all_articles.extend(page_articles)

        if page_number == max_pages:
            print(f"Reached page {max_pages} for {company}.")
            return all_articles, True

        # Attempt next page
        if not click_next_page(driver):
            print(f"🔚 No more pages for {company}.")
            return all_articles, False
        
        page_number += 1

    return all_articles, (page_number == max_pages)

def analyze_sentiment_for_articles(articles):
    """
    Perform sentiment analysis on article headlines using VADER.
    Each article is a dict {headline, date?}, so we'll add 'sentiment'.
    """
    sid = SentimentIntensityAnalyzer()
    for article in articles:
        score = sid.polarity_scores(article["headline"])["compound"]
        article["sentiment"] = score
    return articles


def write_csv_chunk(company, articles, mode="a", write_header=False):
    """
    Append chunk of articles to the CSV. 
    If 'write_header' is True => open file in 'w' mode and write the header.
    Otherwise => open in 'a' mode and skip the header.
    """
    filename = f"{company.replace(' ', '_')}_news_sentiment.csv"
    fieldnames = ["headline", "date", "sentiment"]
    
    if write_header:
        mode = "w"  # Overwrite any existing file and write the header

    with open(filename, mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for art in articles:
            row = {
                "headline": art["headline"],
                "date": art["date"].strftime("%Y-%m-%d") if art["date"] else "",
                "sentiment": art.get("sentiment", "")
            }
            writer.writerow(row)

    print(f"✅ Wrote {len(articles)} articles to {filename} with mode='{mode}'.")

def main():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.5304.87 Safari/537.36"
    )
    service = Service("/Users/wongyule/Documents/Designing Intelligent Agents/FinancialTradingBot/chromedriver-mac-arm64/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)

    companies = [
        "3M Corporation",
        "American Express Company",
    ]

    # For each company
    for company in companies:
        print(f"\n=== Processing {company} ===")
        cd_min_date = datetime(2022, 1, 1)
        cd_max_date = datetime(2024, 12, 31)

        # Start fresh CSV with header
        write_csv_chunk(company, [], mode="w", write_header=True)
        
        while cd_max_date > cd_min_date:
            cd_min_str = cd_min_date.strftime("%m/%d/%Y")
            cd_max_str = cd_max_date.strftime("%m/%d/%Y")
            print(f"🔍 Scraping {company} => {cd_min_str} to {cd_max_str}")

            # Scrape chunk (pages up to 30)
            chunk_articles, reached_30 = scrape_google_news_range(
                company, cd_min_str, cd_max_str, driver, max_pages=30
            )
            
            # If we got any articles, do sentiment + write immediately
            if chunk_articles:
                # Possibly you want to parse date individually if your data structure is consistent
                # For demonstration, let's do a quick sentiment pass
                chunk_articles = analyze_sentiment_for_articles(chunk_articles)
                
                # Append to CSV
                write_csv_chunk(company, chunk_articles, mode="a", write_header=False)

            # If we reached page 30 => subtract 180 days, continue scraping older range
            if reached_30:
                cd_max_date -= timedelta(days=180)
            else:
                # If we didn't fill 30 pages, means no more news => go next company
                print("No more data in this range => moving on to next company.")
                break

    driver.quit()

if __name__ == "__main__":
    main()
