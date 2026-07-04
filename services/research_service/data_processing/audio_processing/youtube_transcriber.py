from dotenv import load_dotenv
import sys
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from services.research_service.data_processing.doc_processing.doc_processor import DocumentChunk
from youtube_transcript_api import YouTubeTranscriptApi
from backend.core.exceptions import CustomException
from backend.core.logging import logging
import config
from urllib.parse import parse_qs, urlparse

load_dotenv()

@dataclass
class TranscriptData:
    """Represents a processed youtube transcript data chunk with metadata for citations"""
    content: str
    metadata: Dict[str, Any]

class YoutubeTranscriber:
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        self.api=YouTubeTranscriptApi()
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        logging.info("YoutubeTranscriber initialized.")
    

    def _extract_video_id(self, url: str) -> Optional[str]:
        try:
            parsed = urlparse(url)
            if parsed.hostname in {"youtu.be", "www.youtu.be"}:
                video_id = parsed.path.lstrip("/")
            elif parsed.hostname and "youtube.com" in parsed.hostname:
                if parsed.path.startswith("/shorts/") or parsed.path.startswith("/embed/"):
                    video_id = parsed.path.split("/")[2]
                else:
                    video_id = parse_qs(parsed.query).get("v", [None])[0]
            else:
                video_id = None
            return video_id
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)  
    

    def _download_transcript(self, url: str) -> str:
        video_id = self._extract_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID from URL")

        try:
            transcript = self.api.fetch(
                video_id,
                languages=("en", "en-US", "en-GB", "hi")
            )
        except Exception:
            transcript_list = self.api.list(video_id)
            transcript = transcript_list.find_transcript(
                [t.language_code for t in transcript_list]
            ).fetch()

        text = " ".join([snippet.text for snippet in transcript.snippets])

        return text


    def _chunk_processed_scraped_data(
        self,
        content: TranscriptData,
        video_id: str,
        url: str)-> List[DocumentChunk]:

        
        if not content.strip():
            return []
        
        chunks = []
        text = content
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                boundary = max(last_period, last_newline)
                if boundary > start + self.chunk_size * 0.5:
                    end = boundary + 1

            chunk_text = text[start:end].strip()

            chunk_metadata = {
                "video_id": video_id,
                "video_url": url,
                "source_url": url
            }


            chunk = DocumentChunk(
                    content=chunk_text,
                    source_file=url,
                    source_type="youtube",
                    page_number=None,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end-1,
                    metadata=chunk_metadata
                )

            chunks.append(chunk)
            chunk_index += 1

            start = max(start + self.chunk_size - self.chunk_overlap, end)

            if start >= len(text):
                break
        
        return chunks
    
    def process_transcript(self, url: str, video_id: str) -> List[DocumentChunk]:
        try:
            transcript = self._download_transcript(url)
            logging.info("Transcript downloaded.")

            chunks = self._chunk_processed_scraped_data(transcript, video_id, url)
            logging.info("Chunks created.")

            return chunks
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error

if __name__ == "__main__":
    
    transcriber = YoutubeTranscriber()
    
    try:
        test_url = "https://www.youtube.com/watch?v=D26sUZ6DHNQ"
        video_id = transcriber._extract_video_id(test_url)
        chunks = transcriber.process_transcript(url=test_url, video_id=video_id)
        
        """ print(f"Transcribed {len(chunks)} utterances:")
        for chunk in chunks[:5]:
            print(f"  {chunk.content}")"""
        print(chunks)
        
    except Exception as e:
        print(f"Error: {e}")
