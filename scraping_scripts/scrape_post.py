import httpx
import json
from typing import Dict
from urllib.parse import quote
from parse_post import parse_post

INSTAGRAM_DOCUMENT_ID = "8845758582119845" # contst id for post documents instagram.com


def scrape_post(url_or_shortcode: str) -> Dict:
    """Scrape single Instagram post data"""
    if "http" in url_or_shortcode:
        shortcode = url_or_shortcode.split("/p/")[-1].split("/")[0]
    else:
        shortcode = url_or_shortcode
    print(f"scraping instagram post: {shortcode}")

    variables = quote(json.dumps({
        'shortcode':shortcode,'fetch_tagged_user_count':None,
        'hoisted_comment_id':None,'hoisted_reply_id':None
    }, separators=(',', ':')))
    body = f"variables={variables}&doc_id={INSTAGRAM_DOCUMENT_ID}"
    url = "https://www.instagram.com/graphql/query"
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
        "Referer": f"https://www.instagram.com/p/{shortcode}/",
        "Origin": "https://www.instagram.com"
    }

    result = httpx.post(
        url=url,
        headers = headers,
        data=body
    )
    data = json.loads(result.content)
    return data["data"]["xdt_shortcode_media"]

# Example usage:    
posts = scrape_post("https://www.instagram.com/p/CuE2WNQs6vH/")
parsed_post = parse_post(posts)

# save a JSON file
with open("result.json", "w",encoding="utf-8") as f:
    json.dump(parsed_post, f, indent=2, ensure_ascii=False)

