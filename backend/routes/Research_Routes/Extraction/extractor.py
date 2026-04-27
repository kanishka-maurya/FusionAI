from backend.routes.Research_Routes.Extraction.github_extractor import extract_github_content
from backend.routes.Research_Routes.Extraction.arxiv_extractor import extract_arxiv_content
from backend.routes.Research_Routes.Extraction.news_extractor import extract_news_content
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

async def extract_full_document(item):
    source = item["source"]
    url = item["url"]

    if source == "github":
        content = await extract_github_content(url)

    elif source == "papers":
        content = await extract_arxiv_content(url)

    elif source == "news":
        content = await extract_news_content(url)

    else:
        content = ""

    return clean_text(content)