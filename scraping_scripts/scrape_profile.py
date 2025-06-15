import json
import httpx
from urllib.parse import quote
from typing import Optional
import sys
import os

# Thoughts / TODO
# - scrape how many followers a page has and scale each like_count by followers
# - 

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import USERNAME
INSTAGRAM_ACCOUNT_DOCUMENT_ID = "9310670392322965"

def scrape_follower_count(username: str) -> int:
    """Scrape follower count for the given username."""

    client = httpx.Client(
        headers={
            # this is internal ID of an instegram backend app. It doesn't change often.
            "x-ig-app-id": "936619743392459",
            # use browser-like features
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "*/*",
        }
    )

    result = client.get(f"https://i.instagram.com/api/v1/users/web_profile_info/?username={username}",)
    data = json.loads(result.content)
    return data["data"]["user"]["edge_followed_by"]["count"]


async def scrape_user_posts(username: str, page_size=12, max_pages: Optional[int] = None):
    """Scrape all posts of an Instagram user given the username."""
    base_url = "https://www.instagram.com/graphql/query"
    variables = {
        "after": None,
        "before": None,
        "data": {
            "count": page_size,
            "include_reel_media_seen_timestamp": True,
            "include_relationship_info": True,
            "latest_besties_reel_media": True,
            "latest_reel_media": True
        },
        "first": page_size,
        "last": None,
        "username": f"{username}",
        "__relay_internal__pv__PolarisIsLoggedInrelayprovider": True,
        "__relay_internal__pv__PolarisShareSheetV3relayprovider": True
    }

    prev_cursor = None
    _page_number = 1

    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as session:
        while True:
            body = f"variables={quote(json.dumps(variables, separators=(',', ':')))}&doc_id={INSTAGRAM_ACCOUNT_DOCUMENT_ID}"
            cookies = {
                "sessionid": "75247192591%3Ag9zb9M4P9xrfUy%3A20%3AAYe3qr32CLFMJDGBKx7Sv0Q6tYK3ZLHJUPR4H93D4w",
                "csrftoken": "pdeCawUfZesM4IixqjVTNe6MZ0YKsCej",
            }

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "X-CSRFToken": cookies["csrftoken"],
                "X-Instagram-AJAX": "1007882423", 
                "X-Requested-With": "XMLHttpRequest",
                "Origin": "https://www.instagram.com"
            }
            response = await session.post(
                base_url,
                data=body,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            posts = data["data"]["xdt_api__v1__feed__user_timeline_graphql_connection"]
            for post in posts["edges"]:
                yield post["node"]

            page_info = posts["page_info"]
            if not page_info["has_next_page"]:
                print(f"scraping posts page {_page_number}")
                break

            if page_info["end_cursor"] == prev_cursor:
                print("found no new posts, breaking")
                break

            prev_cursor = page_info["end_cursor"]
            variables["after"] = page_info["end_cursor"]
            _page_number += 1




# Example run:
if __name__ == "__main__":
    import asyncio

    async def main(username):
        posts = [post async for post in scrape_user_posts(username, page_size=12, max_pages=50)]
        follower_count = scrape_follower_count(username)
        data_to_save = {
            "_metadata": {
                "username": username,
                "follower_count": follower_count
            },
            "posts": posts
        }
        with open(f"data/{username}.json", "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    asyncio.run(main(USERNAME))


def run_scraper(username):
    """Synchronous wrapper for the async scraper that also saves follower count."""
    import asyncio
    
    async def scrape_and_save(username):
        print(f"Scraping posts for {username}...")
        posts = [post async for post in scrape_user_posts(username, page_size=12, max_pages=50)]
        
        print(f"Scraping follower count for {username}...")
        follower_count = scrape_follower_count(username)

        data_to_save = {
            "_metadata": {
                "username": username,
                "follower_count": follower_count
            },
            "posts": posts
        }

        # Ensure directory exists
        os.makedirs("data", exist_ok=True)
        with open(f"data/{username}.json", "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(posts)} posts and follower count = {follower_count:,} for {username}")
    
    asyncio.run(scrape_and_save(username))

