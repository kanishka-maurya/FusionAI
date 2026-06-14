import httpx
import io
from pdfminer.high_level import extract_text

async def extract_arxiv_content(paper_url):
    try:
        pdf_url = paper_url.replace("abs", "pdf") + ".pdf"

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            res = await client.get(pdf_url)
            res.raise_for_status()

        pdf_file = io.BytesIO(res.content)

        text = extract_text(pdf_file)

        return text[:5000]  # limit for now
    except Exception as exc:
        print(f"[ARXIV EXTRACTION ERROR] {paper_url}: {exc}")
        return ""
