import httpx
from bs4 import BeautifulSoup

async def extract_news_content(url):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url)

        soup = BeautifulSoup(res.text, "html.parser")

        paragraphs = soup.find_all("p")

        text = " ".join([p.get_text() for p in paragraphs])

        return text[:5000]
    except:
        return ""