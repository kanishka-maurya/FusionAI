import httpx
import io
from pdfminer.high_level import extract_text

async def extract_arxiv_content(paper_url):
    try:
        pdf_url = paper_url.replace("abs", "pdf") + ".pdf"

        async with httpx.AsyncClient() as client:
            res = await client.get(pdf_url)

        pdf_file = io.BytesIO(res.content)

        text = extract_text(pdf_file)

        return text[:5000]  # limit for now
    except:
        return ""