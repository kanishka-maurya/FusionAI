from fastapi import APIRouter, HTTPException,Request, Header
from pathlib import Path
from urllib.parse import urlparse
import os

from services.research_service.embeddings.embedding_generator import EmbeddingGenerator
from services.research_service.vector_database.vector_database import ChromaVectorDatabase
from services.research_service.data_processing.web_scraping.web_scraper import WebScraper
from services.research_service.generation.generation import RAGGenerator, RAGResult
from memory.memory import NotebookMemoryLayer

router = APIRouter()
embedding_generator = EmbeddingGenerator()
vector_db = ChromaVectorDatabase()

UPLOAD_DIR = Path("web_upload")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/web_upload")
async def upload_url(url: str,
                      request:Request):
    try:
        # Validate API key (like a guard, not exit)
        print(url)
        api_key = os.getenv("FIRECRAWL_API_KEY")
        web_scraper = WebScraper(api_key=api_key)
        user_id = getattr(request.state, "user_id", None)
        session_id = getattr(request.state, "notebook_id", None)
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="FIRECRAWL_API_KEY environment variable not set"
            )

        # Validate URL format
        try:
          parsed_url = urlparse(url)
          if not parsed_url.scheme or not parsed_url.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL")
        except Exception as e:
            print(e)
        # Scrape content
        chunks = web_scraper.scrape_url(url)

        # Generate embeddings
        embedded_chunks = embedding_generator.generate_embeddings(chunks)

        print(f"Generated {len(embedded_chunks)} embeddings")
        print(embedded_chunks)

        # Insert into vector DB
        inserted_ids = vector_db.insert_embeddings(embedded_chunks,  user_id=user_id, session_id=session_id)

        print(f"Inserted {len(inserted_ids)} embeddings")

        return {
            "url": url,
            "total_chunks": len(chunks),
            "chunks_preview": [
                {
                    "content": chunk.content[:200],
                    "page": chunk.page_number,
                    "chunk_id": chunk.chunk_id
                }
                for chunk in chunks[:5]
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

