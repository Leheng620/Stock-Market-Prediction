from newsplease import NewsPlease
from GoogleNews import GoogleNews
import os
from datetime import datetime

def search_news_urls(stock_symbols, num_articles=5):
    gn = GoogleNews(lang='en', period='7d')
    news_urls = {}

    for stock in stock_symbols:
        gn.search(stock)
        results = gn.results(sort=True)[:num_articles]
        news_urls[stock] = [result["link"] for result in results]

    return news_urls

def fetch_and_save_articles(stock_urls, directory="stock_news"):
    os.makedirs(directory, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")

    for stock, urls in stock_urls.items():
        for url in urls:
            try:
                article = NewsPlease.from_url(url)
                file_name = f"{stock}_{article.title}_{date_str}.txt".replace(" ", "_").replace("/", "-")
                file_path = os.path.join(directory, file_name)

                with open(file_path, "w") as f:
                    f.write(f"Title: {article.title}\n")
                    f.write(f"Author: {', '.join(article.authors)}\n")
                    f.write(f"Date: {article.date_publish}\n")
                    f.write(f"URL: {url}\n\n")
                    f.write(article.maintext)

            except Exception as e:
                print(f"Error fetching article from {url}: {e}")

if __name__ == "__main__":
    # Replace the stock symbols with the ones you are interested in
    stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    # Get news URLs related to stock symbols
    stock_urls = search_news_urls(stock_symbols)

    # Fetch and save news articles
    fetch_and_save_articles(stock_urls)
