import logging
from typing import List, Dict, Any, Optional
import json
import chromadb
from chromadb.config import Settings
import sys
from backend.core.exceptions import CustomException

from services.research_service.embeddings.embedding_generator import EmbeddedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaVectorDatabase:
    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "fusionai_collection"
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=self.db_path)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        logger.info(f"ChromaDB initialized at {self.db_path}")

    def insert_embeddings(self, embedded_chunks: List[EmbeddedChunk]) -> List[str]:
        if not embedded_chunks:
            return []

        try:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            print(embedded_chunks[0])
            for chunk in embedded_chunks:
                data = chunk.to_vector_db_format()

                ids.append(data["id"])
                documents.append(data["content"])
                embeddings.append(data["vector"])

                metadata = {
                    "source_file": data.get("source_file"),
                    "source_type": data.get("source_type"),
                    "page_number": data.get("page_number", -1),
                    "chunk_index": data.get("chunk_index"),
                    "start_char": data.get("start_char", -1),
                    "end_char": data.get("end_char", -1),
                    "embedding_model": data.get("embedding_model"),
                }

        
                if isinstance(data.get("metadata"), dict):
                    metadata.update(data["metadata"])

                clean_metadata = {}

                for key, value in metadata.items():
                  if value is None:
                    continue
                  elif isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                  else:
                    clean_metadata[key] = str(value)

                metadatas.append(clean_metadata)

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

            logging.info(f"Inserted {len(ids)} embeddings")
            return ids

        except Exception as e:
                error = CustomException(e, sys)
                logging.error(error)
                raise error

    def search(
        self,
        query_vector: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:

        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit
            )
            logging.info("Vector search concluded....")
            formatted_results = []

            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]

                formatted_results.append({
                    "id": results["ids"][0][i],
                    "score": results["distances"][0][i],
                    "content": results["documents"][0][i],
                    "citation": {
                        "source_file": metadata.get("source_file"),
                        "source_type": metadata.get("source_type"),
                        "page_number": metadata.get("page_number"),
                        "chunk_index": metadata.get("chunk_index"),
                        "start_char": metadata.get("start_char"),
                        "end_char": metadata.get("end_char"),
                    },
                    "metadata": metadata,
                    "embedding_model": metadata.get("embedding_model")
                })
            logging.info(f"Search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
                error = CustomException(e, sys)
                logging.error(error)
                raise error


    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        try:
            result = self.collection.get(ids=[chunk_id])

            if not result["ids"]:
                return None

            metadata = result["metadatas"][0]

            return {
                "id": result["ids"][0],
                "content": result["documents"][0],
                "metadata": metadata,
                "source_file": metadata.get("source_file"),
                "source_type": metadata.get("source_type"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
            }

        except Exception as e:
            logger.error(f"Get by ID error: {str(e)}")
            return None

   
    def delete_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            logger.error(f"Delete error: {str(e)}")
            raise

    def close(self):
        try:
            self.client.persist()
            logger.info("ChromaDB persisted successfully")
        except Exception as e:
            logger.error(f"Close error: {str(e)}")

if __name__ == "__main__":
    from services.research_service.data_processing.doc_processing.doc_processor import DocumentProcessor
    from services.research_service.embeddings.embedding_generator import EmbeddingGenerator
    
    doc_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()
    
    vector_db = ChromaVectorDatabase()
    
    try:
        chunks = doc_processor.process_document(r"C:\Users\kanis\FusionAI\services\research_service\vector_database\CRAG Paper.pdf")
        print(chunks)
        embedded_chunks = embedding_generator.generate_embeddings(chunks)
        print(len(embedded_chunks))
        print(embedded_chunks)

        inserted_ids = vector_db.insert_embeddings(embedded_chunks)
        print(f"Inserted {len(inserted_ids)} embeddings")
        
        query_text = "What is the main topic?"
        query_vector = embedding_generator.generate_query_embedding(query_text)
        
        search_results = vector_db.search(
            query_vector.tolist(), 
            limit=5
        )
        
        for i, result in enumerate(search_results):
            print(f"\nResult {i+1}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Content: {result['content'][:200]}...")
            print(f"Citation: {result['citation']}")
    
        if inserted_ids:
            sample_id = inserted_ids[0]
            chunk = vector_db.get_chunk_by_id(sample_id)
            print("\nSample chunk fetch:")
            print(chunk)
        
    except Exception as e:
        print(f"Error in example: {e}")
    
    finally:
        vector_db.close()