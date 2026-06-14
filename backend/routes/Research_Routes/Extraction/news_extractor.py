import httpx
from bs4 import BeautifulSoup

async def extract_news_content(url):
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            res = await client.get(url)
            res.raise_for_status()

        soup = BeautifulSoup(res.text, "html.parser")

        paragraphs = soup.find_all("p")

        text = " ".join([p.get_text() for p in paragraphs])

        return text[:5000]
    except Exception as exc:
        print(f"[NEWS EXTRACTION ERROR] {url}: {exc}")
        return ""
