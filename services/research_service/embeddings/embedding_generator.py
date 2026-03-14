from dataclasses import dataclass
from services.research_service.data_processing.doc_processing.doc_processor import DocumentChunk
from typing import List, Dict, Any, Tuple
from backend.core.exceptions import CustomException
from backend.core.logging import logging
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import sys
from dotenv import load_dotenv

load_dotenv()

@dataclass 
class EmbeddedChunk:
    """Document chunk with its embedding vector"""
    chunk: DocumentChunk
    embedding: np.ndarray
    embedding_model: str

    def to_vector_db_format(self) -> Dict[str, Any]:
        return {
            'id': self.chunk.chunk_id,
            'vector': self.embedding.tolist(),
            'content': self.chunk.content,
            'source_file': self.chunk.source_file,
            'source_type': self.chunk.source_type,
            'page_number': self.chunk.page_number,
            'chunk_index': self.chunk.chunk_index,
            'start_char': self.chunk.start_char,
            'end_char': self.chunk.end_char,
            'metadata': self.chunk.metadata,
            'embedding_model': self.embedding_model
        } 
    
class EmbeddingGenerator:
    def __init__(self, model_name: str="gemini-embedding-001"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            logging.info("Initializing embedding model.")
            self.model = GoogleGenerativeAIEmbeddings(model=self.model_name)

            sample_embedding = [self.model.embed_query(["test"])][0]
            self.embedding_dim = len(sample_embedding)

            logging.info(f"Model initialized successfully. Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error
    
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[EmbeddedChunk]:
        if not chunks:
            return []
        
        logging.info(f"Generating embeddings for {len(chunks)} chunks")
        
        try:
            texts = [chunk.content for chunk in chunks]
            
            embeddings = list(self.model.embed_query(texts))
            embedded_chunks = []
            for chunk, embedding in zip(chunks, embeddings):
                embedded_chunk = EmbeddedChunk(
                    chunk=chunk,
                    embedding=np.array(embedding, dtype=np.float32),
                    embedding_model=self.model_name
                )
                embedded_chunks.append(embedded_chunk)
            
            logging.info(f"Successfully generated {len(embedded_chunks)} embeddings")
            return embedded_chunks
            
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error
    
    def generate_query_embedding(self, query_text: str) -> np.ndarray:
        try:
            embedding = list(self.model.embed_query([query_text]))[0]
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
    
    def batch_generate_embeddings(
        self, 
        chunks_batches: List[List[DocumentChunk]], 
        batch_size: int = 32
    ) -> List[List[EmbeddedChunk]]:
        
        all_embedded_batches = []
        for i, chunk_batch in enumerate(chunks_batches):
            logging.info(f"Processing batch {i+1}/{len(chunks_batches)}")
            
            embedded_batch = []
            for j in range(0, len(chunk_batch), batch_size):
                sub_batch = chunk_batch[j:j + batch_size]
                embedded_sub_batch = self.generate_embeddings(sub_batch)
                embedded_batch.extend(embedded_sub_batch)
            
            all_embedded_batches.append(embedded_batch)
            
        return all_embedded_batches


if __name__ == "__main__":
    from services.research_service.data_processing.doc_processing.doc_processor import DocumentProcessor
    
    doc_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()
    
    try:
        chunks = doc_processor.process_document(r"C:\Users\kanis\FusionAI\services\research_service\embeddings\CRAG Paper.pdf")
        embedded_chunks = embedding_generator.generate_embeddings(chunks)
        
        if embedded_chunks:
            sample = embedded_chunks[0]
            print(f"Sample embedding shape: {sample.embedding.shape}")
            print(f"Sample content: {sample.chunk.content[:500]}...")
            print(f"Citation info: {sample.chunk.get_citation_info()}")
            
            query = "What is the main topic?"
            query_embedding = embedding_generator.generate_query_embedding(query)
            print(f"Query embedding shape: {query_embedding.shape}")
            
    except Exception as e:
        print(f"Error in example usage: {e}")