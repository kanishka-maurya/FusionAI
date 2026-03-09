import os
import sys
from datetime import datetime
import hashlib
from typing import List, Any, Optional, Dict
from pathlib import  Path
import pymupdf
from dataclasses import dataclass
from backend.core.exceptions import CustomException
from backend.core.logging import logging
import config

@dataclass
class DocumentChunk:
    """Represents a processed document chunk with metadata for citations"""
    content: str
    source_file: str
    source_type: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = None
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = self._generate_chunk_id()
        if self.metadata is None:
            self.metadata = {}

    def _generate_chunk_id(self) -> str:
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.source_type}_{self.chunk_index}_{content_hash}"
    
    def get_citation_info(self) -> Dict[Dict, Any]:
        citation = {
            'source': self.source_file,
            'type': self.source_type,
            'chunk_id': self.chunk_id,
            'chunk_index': self.chunk_index
        }

        if self.page_number:
            citation["page_number"] = self.page_number
        if self.start_char or self.end_char:
            citation['char_range'] = f"{self.start_char}-{self.end_char}"
        
        citation.update(self.metadata)
        return citation
    
class DocumentProcessor:
    def __init__(self, chunk_size:int=config.CHUNK_SIZE, chunk_overlap:int=config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = config.SUPPORTED_FORMATS


    def _create_chunks_from_text(
                self, 
                text: str, 
                source_file: str, 
                source_type: str,
                page_number: Optional[int] = None,
                additional_metadata: Dict[str, Any] = None
                ) -> List[DocumentChunk]:

        if not text.strip():
            return []
        
        chunks = []
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

            if chunk_text:
                chunk_metadata = additional_metadata.copy() if additional_metadata else {}

                chunk = DocumentChunk(
                    content=chunk_text,
                    source_file=source_file,
                    source_type=source_type,
                    page_number=page_number,
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
    

    def _process_pdf(self, file_path: Path) -> List[DocumentChunk]:
        chunks = []
        try:
            doc = pymupdf.open(file_path)
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if not text.strip():
                    continue
                
                # Get page metadata
                page_metadata = {
                    'total_pages': total_pages,
                    'page_width': page.rect.width,
                    'page_height': page.rect.height,
                    'processed_at': datetime.now().isoformat()
                }
                
                page_chunks = self._create_chunks_from_text(
                    text, 
                    file_path.name, 
                    source_type='pdf', 
                    page_number=page_num+1,
                    additional_metadata=page_metadata
                )
            
                chunks.extend(page_chunks)
            doc.close()
            logging.info(f"Processed PDF {file_path.name}: {len(chunks)} chunks from {total_pages} pages.")
            
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error

        return chunks


    def _process_text_file(self, file_path: Path) -> List[DocumentChunk]:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            metadata = {
                'file_size': file_path.stat().st_size,
                'encoding': 'utf-8',
                'processed_at': datetime.now().isoformat()
            }
            
            chunks = self._create_chunks_from_text(
                content, 
                file_path.name, 
                source_type='txt', 
                page_number=None,
                additional_metadata=metadata
            )
            
            logging.info(f"Processed text file {file_path.name}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error


    def process_document(self, file_path: Path) -> List[DocumentChunk]:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in self.supported_formats:
            logging.error(f"Unsupported file format: {file_path.suffix}")
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logging.info(f"Processing document: {file_path.name}")

        try:
            if file_path.suffix.lower() == '.pdf':
                return self._process_pdf(file_path)
            elif file_path.suffix.lower() in {'.txt', '.md'}:
                return self._process_text_file(file_path)
                
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error
        

    def batch_process(self, file_paths: List[str]) -> List[DocumentChunk]:
        all_chunks = []
        for file_path in file_paths:
            try:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)

            except Exception as e:
                error = CustomException(e, sys)
                logging.error(error)
                continue
        
        logging.info(f"Batch processing complete: {len(all_chunks)} total chunks from {len(file_paths)} files")
        return all_chunks
        

    

    








