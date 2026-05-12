from dataclasses import dataclass
from services.research_service.data_processing.doc_processing.doc_processor import DocumentChunk
from typing import List, Dict, Any
from backend.core.exceptions import CustomException
from backend.core.logging import logging
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import sys
import time
import re
from dotenv import load_dotenv

load_dotenv()

@dataclass 
class EmbeddedChunk:
    """Document chunk with its embedding vector"""
    chunk: DocumentChunk
    embedding: np.ndarray
    embedding_model: str

    def to_vector_db_format(self, user_id, session_id) -> Dict[str, Any]:
    # Ensure metadata exists
      metadata = self.chunk.metadata if self.chunk.metadata else {}
    
    # FORCE the IDs into the metadata so they aren't lost
      metadata["user_id"] = str(user_id)
      metadata["session_id"] = str(session_id)

      return {
        "user_id": user_id,
        "session_id": session_id,
        'id': self.chunk.chunk_id,
        'vector': self.embedding.tolist(),
        'content': self.chunk.content,
        'source_file': self.chunk.source_file,
        'source_type': self.chunk.source_type,
        'page_number': self.chunk.page_number,
        'chunk_index': self.chunk.chunk_index,
        'metadata': metadata, # Use the updated metadata dict
        'embedding_model': self.embedding_model
    }


class EmbeddingGenerator:
    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None

        self.max_batch_size = 100 
        self.retry_attempts = 3
        self.base_retry_delay = 2 
        self.min_request_interval = 0.7  
        self.last_request_time = 0

        self._initialize_model()


    def _initialize_model(self):
        try:
            logging.info("Initializing embedding model.")
            self.model = GoogleGenerativeAIEmbeddings(model=self.model_name)

            logging.info(f"Model initialized.")

        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error


    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[EmbeddedChunk]:
        if not chunks:
            return []

        logging.info(f"Generating embeddings for {len(chunks)} chunks")

        try:
            embedded_chunks = []

            for i in range(0, len(chunks), self.max_batch_size):
                batch_chunks = chunks[i:i + self.max_batch_size]
                texts = [chunk.content for chunk in batch_chunks]

                self._throttle_requests()

                embeddings = self._embed_with_retry(texts)

                for chunk, embedding in zip(batch_chunks, embeddings):
                    embedded_chunks.append(
                        EmbeddedChunk(
                            chunk=chunk,
                            embedding=np.array(embedding, dtype=np.float32),
                            embedding_model=self.model_name
                        )
                    )

            logging.info(f"Generated {len(embedded_chunks)} embeddings")
            print("embedded chunks",embedded_chunks)
            return embedded_chunks
            
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error
        
    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(self.retry_attempts):
            try:
                return self.model.embed_documents(texts)

            except Exception as e:
                error_msg = str(e)
                logging.warning(f"Embedding failed (attempt {attempt+1}): {error_msg}")

                
                if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                    wait_time = self._extract_retry_delay(error_msg)
                    logging.warning(f"Quota exceeded. Waiting {wait_time}s...")
                    time.sleep(wait_time)

                else:
                    
                    time.sleep(self.base_retry_delay * (attempt + 1))

                if attempt == self.retry_attempts - 1:
                    raise
                
    def _extract_retry_delay(self, error_msg: str) -> int:
        match = re.search(r"retry in (\d+\.?\d*)s", error_msg.lower())
        if match:
            return int(float(match.group(1))) + 1
        return 30  
    def _throttle_requests(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        self.last_request_time = time.time()

    def generate_query_embedding(self, query_text: str) -> np.ndarray:
        try:
            self._throttle_requests()
            embedding = self.model.embed_query(query_text)
            logging.info("Query embedding is generated....")
            return np.array(embedding, dtype=np.float32)
            

        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

