import requests
import json
import os

from bs4 import BeautifulSoup
import pandas as pd
from dotenv import load_dotenv
from geopy.extra.rate_limiter import RateLimiter

session = requests.Session()


def throttle(min_delay):
    def decorator(func):
        rate_limiter = RateLimiter(func, min_delay_seconds=min_delay)
        def wrapper(*args, **kwargs):
            return rate_limiter(*args, **kwargs)
        return wrapper
    return decorator


def save(data, filename):
    os.makedirs("data", exist_ok=True)
    with open(f"data/{filename}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("Data collection complete. Saved to 'data/' directory.")


# -------------- GitHub Scraper -----------------
load_dotenv()
HEADERS = {"Authorization": f"token {os.getenv('github_key')}"}
GITHUB_QUERY = "language:Stata+extension:do"  # Search Stata files


@throttle(1)
def search_github_stata_files():
    """Search GitHub for Stata repositories containing .do files"""
    url = f"https://api.github.com/search/code?q={GITHUB_QUERY}&sort=stars&per_page=10"
    response = session.get(url, headers=HEADERS)

    stata_files = []
    if response.status_code == 200:
        for file in response.json()["items"]:
            file_info = requests.get(file["url"], headers=HEADERS).json()
            file_content = requests.get(file_info["download_url"], headers=HEADERS).text
            stata_files.append({"filename": file["name"], "content": file_content})
        return stata_files
    else:
        print("Error fetching repositories:", response.json())
        return []


# -------------- Statalist Scraper -----------------
STATLIST_URL = "https://www.statalist.org/forums/forum/general-stata-discussion/general"
BASE_URL = "https://www.statalist.org"


def scrape_statalist():
    """Scrape Statalist for discussions containing Stata code examples"""
    response = requests.get(STATLIST_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    discussions = []
    for thread in soup.select(".title a"):
        thread_url = BASE_URL + thread["href"]
        thread_title = thread.get_text(strip=True)

        thread_page = requests.get(thread_url)
        thread_soup = BeautifulSoup(thread_page.text, "html.parser")

        posts = thread_soup.select(".content")
        thread_content = "\n".join([p.get_text(strip=True) for p in posts])

        discussions.append({"title": thread_title, "url": thread_url, "content": thread_content})

    return discussions


# -------------- Main Functions -----------------
def github_pull():
    print("Scraping GitHub for Stata repositories...")
    files = search_github_stata_files()
    save(files, "github_repos")


def statalist_pull():
    print("Scraping Statalist for discussions...")
    statalist_data = scrape_statalist()
    save(statalist_data, "statalist_discussions")


if __name__ == "__main__":
    github_pull()
