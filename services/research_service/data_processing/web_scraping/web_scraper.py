import os
import sys
from datetime import datetime
from typing import List, Any, Optional, Dict
import time
from dataclasses import dataclass
from backend.core.exceptions import CustomException
from backend.core.logging import logging
import config
from firecrawl import Firecrawl
from urllib.parse import urlparse
from dotenv import load_dotenv
from services.research_service.data_processing.doc_processing.doc_processor import DocumentChunk

load_dotenv()

@dataclass
class WebPageData:
    """Represents a processed web scraped data chunk with metadata for citations"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class WebScraper:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.app = Firecrawl(api_key=self.api_key)
        logging.info("WebScraper initialized with firecrawl.")


    def _is_valid_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    
    def _process_scraped_data(self, result: Dict[str, Any], url: str) -> WebPageData:
        try:
            content = getattr(result, "markdown", "")
            metadata_dict = getattr(result, "metadata_dict", {}) or {}
            metadata = {
                    'scraped_at': datetime.now().isoformat(),
                    'original_url': url,
                    'title': metadata_dict.get('title', ''),
                    'description': metadata_dict.get('description', ''),
                    'keywords': metadata_dict.get('keywords', []),
                    'language': metadata_dict.get('language', 'en'),
                    'word_count': len(content.split()) if content else 0,
                    'character_count': len(content) if content else 0,
                    'domain': urlparse(url).netloc
                }
            return WebPageData(
                url=url,
                title=metadata["title"] or f"Web Page - {metadata['domain']}",
                content=content,
                metadata=metadata,
                success=True
            )
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error


    def _chunk_processed_scraped_data(
        self,
        page_data: WebPageData,
        chunk_size: int,
        chunk_overlap: int
    )-> List[DocumentChunk]:
        
        if not page_data.success or not page_data.content.strip():
            logging.warning(f"No content to process for {page_data.url}")
            return []
        
        chunks = []
        content = page_data.content
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            if end < len(content):
                last_double_newline = content.rfind('\n\n', start, end)
                if last_double_newline > start + chunk_size * 0.3:
                    end = last_double_newline + 2
                else:
                    last_period = content.rfind('.', start, end)
                    if last_period > start + chunk_size * 0.5:
                        end = last_period + 1
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunk_metadata = page_data.metadata.copy()
                chunk_metadata.update({
                    'chunk_character_start': start,
                    'chunk_character_end': end - 1,
                    'url_fragment': f"{page_data.url}#chunk-{chunk_index}"
                })
                
                chunk = DocumentChunk(
                    content=chunk_text,
                    source_file=page_data.title,
                    source_type='web',
                    page_number=None,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end-1,
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            start = max(start + chunk_size - chunk_overlap, end)
        
        return chunks        
    

    def scrape_url(self, url: str, wait_for_results: int = 30, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP)-> List[DocumentChunk]:
        """
        WebScraper handles extracting and processing web content for AI
        research and RAG pipelines.

        It uses the Firecrawl API to scrape webpages, extracts text and
        metadata, and splits the content into structured chunks suitable
        for embedding and retrieval.

        The class ensures each chunk includes source metadata such as
        URL, title, and position information to support traceable
        citations in downstream AI systems.
        """
        
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL format: {url}")
        logging.info(f"Scraping URL: {url}")
        try: 
            scrape_params = {
                'formats': ['markdown', 'html'],
                'timeout': wait_for_results * 1000
            }

            # scraping
            result = self.app.scrape(url, **scrape_params)
            page_data = self._process_scraped_data(result, url)

            # chunking
            chunks = self._chunk_processed_scraped_data(
                    page_data, 
                    chunk_size, 
                    chunk_overlap
                )
            logging.info(f"Successfully scraped {url}: {len(chunks)} chunks created")
            return chunks
        
        except Exception as e:
            logging.error(f"Error scraping URL {url}: {str(e)}")
            raise 


    def batch_scrape_urls(
        self,
        urls: List[str],
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        delay_between_requests: float = 1.0
        ) -> List[List[DocumentChunk]]:
        
        all_chunks = []
        for i, url in enumerate(urls):
            try:
                chunks = self.scrape_url(url=url, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                all_chunks.append(chunks)
                logging.info(f"Successfully scraped {url}: {len(chunks)} chunks")
                
                if i < len(urls) - 1:
                    time.sleep(delay_between_requests)
                    
            except Exception as e:
                logging.error(f"Failed to scrape {url}: {str(e)}")
                all_chunks.append([])
        
        total_chunks = sum(len(chunks) for chunks in all_chunks)
        logging.info(f"Batch scraping complete: {total_chunks} total chunks from {len(urls)} URLs")
        
        return all_chunks

    
if __name__ == "__main__":
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        print("Please set FIRECRAWL_API_KEY environment variable")
        exit(1)
    
    scraper = WebScraper(api_key)
    
    try:
        test_url = "https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag"
        
        chunks = scraper.scrape_url(test_url)
        print(f"\nScraping Results:")
        print(f"Generated {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"Content: {chunk.content[:200]}...")
            print(f"Source: {chunk.source_file}")
            print(f"URL: {chunk.metadata.get('original_url', 'N/A')}")
            print(f"Citation: [Source: {chunk.source_file}, Type: Web]")
        
        urls = ["https://example.com/page1", "https://example.com/page2"]
        batch_results = scraper.batch_scrape_urls(urls)
        
        total_chunks = sum(len(chunks) for chunks in batch_results)
        print(f"\nBatch Results: {total_chunks} total chunks from {len(urls)} URLs")
        
    except Exception as e:
        print(f"Error in scraping example: {e}")