import logging
from typing import List, Dict, Any, Optional
import json
import chromadb
from chromadb.config import Settings

from services.research_service.embeddings.embedding_generator import EmbeddedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorDatabase:
    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "fusionai_collection",
        embedding_dim: Optional[int] = None  # ✅ AUTO-DETECT
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = None
        self.collection = None

        self._initialize_client()
        self._setup_collection()

    # -------------------------------
    # Initialize Client
    # -------------------------------
    def _initialize_client(self):
        try:
            self.client = chromadb.Client(
                Settings(
                    persist_directory=self.db_path,
                    anonymized_telemetry=False
                )
            )
            logger.info(f"ChromaDB initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    # -------------------------------
    # Setup Collection
    # -------------------------------
    def _setup_collection(self):
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            logger.info(f"Collection '{self.collection_name}' ready")

        except Exception as e:
            logger.error(f"Error setting up collection: {str(e)}")
            raise

    # -------------------------------
    # Validate Embedding
    # -------------------------------
    def _validate_embedding(self, vector, chunk_id):
        if not isinstance(vector, list):
            raise ValueError(f"Embedding must be list for {chunk_id}, got {type(vector)}")

        if len(vector) < 10:
            raise ValueError(f"Embedding collapsed for {chunk_id}: {vector}")

        # ✅ Auto-set dimension on first insert
        if self.embedding_dim is None:
            self.embedding_dim = len(vector)
            logger.info(f"Auto-detected embedding dimension: {self.embedding_dim}")

        # ❌ Strict check after that
        if len(vector) != self.embedding_dim:
            raise ValueError(
                f"Dimension mismatch for {chunk_id}: "
                f"{len(vector)} != {self.embedding_dim}"
            )

    # -------------------------------
    # Insert Embeddings
    # -------------------------------
    def insert_embeddings(self, embedded_chunks: List[EmbeddedChunk]) -> List[str]:
        if not embedded_chunks:
            return []

        try:
            ids, documents, embeddings, metadatas = [], [], [], []

            for chunk in embedded_chunks:
                chunk_data = chunk.to_vector_db_format()

                vector = chunk_data["vector"]

                # ✅ CRITICAL VALIDATION
                self._validate_embedding(vector, chunk_data["id"])

                # Handle nulls
                page_number = chunk_data.get("page_number") or -1
                start_char = chunk_data.get("start_char") or -1
                end_char = chunk_data.get("end_char") or -1

                metadata = chunk_data.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}

                # ✅ Flatten metadata
                flat_metadata = {
                    "source_file": chunk_data.get("source_file"),
                    "source_type": chunk_data.get("source_type"),
                    "page_number": page_number,
                    "chunk_index": chunk_data.get("chunk_index"),
                    "start_char": start_char,
                    "end_char": end_char,
                    "embedding_model": chunk_data.get("embedding_model"),
                    **metadata
                }

                ids.append(chunk_data["id"])
                documents.append(chunk_data["content"])
                embeddings.append(vector)
                metadatas.append(flat_metadata)

            # 🔍 Debug once
            logger.info(f"Sample embedding dimension: {len(embeddings[0])}")

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

            logger.info(f"Inserted {len(ids)} embeddings into ChromaDB")
            return ids

        except Exception as e:
            logger.error(f"Error inserting embeddings: {str(e)}")
            raise

    # -------------------------------
    # Search
    # -------------------------------
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:

        try:
            # ✅ Validate query too
            self._validate_embedding(query_vector, "query_vector")

            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=filter_dict
            )

            formatted_results = []

            if results and results["ids"]:
                for i in range(len(results["ids"][0])):
                    metadata = results["metadatas"][0][i]

                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "score": results["distances"][0][i],
                        "content": results["documents"][0][i],
                        "citation": {
                            "source_file": metadata.get("source_file"),
                            "source_type": metadata.get("source_type"),
                            "page_number": None if metadata.get("page_number") == -1 else metadata.get("page_number"),
                            "chunk_index": metadata.get("chunk_index"),
                            "start_char": None if metadata.get("start_char") == -1 else metadata.get("start_char"),
                            "end_char": None if metadata.get("end_char") == -1 else metadata.get("end_char"),
                        },
                        "metadata": metadata,
                        "embedding_model": metadata.get("embedding_model")
                    })

            logger.info(f"Search completed: {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    # -------------------------------
    # Get Chunk by ID
    # -------------------------------
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        try:
            results = self.collection.get(ids=[chunk_id])

            if results and results["ids"]:
                metadata = results["metadatas"][0]

                return {
                    "id": results["ids"][0],
                    "content": results["documents"][0],
                    "metadata": metadata,
                    "source_file": metadata.get("source_file"),
                    "source_type": metadata.get("source_type"),
                    "page_number": metadata.get("page_number"),
                    "chunk_index": metadata.get("chunk_index")
                }

            return None

        except Exception as e:
            logger.error(f"Error retrieving chunk: {str(e)}")
            return None

    # -------------------------------
    # Reset Collection (🔥 VERY USEFUL)
    # -------------------------------
    def reset_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            logger.warning(f"Collection '{self.collection_name}' reset")

            self._setup_collection()

        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise

    # -------------------------------
    # Persist
    # -------------------------------
    def persist(self):
        try:
            self.client.persist()
            logger.info("ChromaDB persisted successfully")
        except Exception as e:
            logger.error(f"Error persisting DB: {str(e)}")

    # -------------------------------
    # Close
    # -------------------------------
    def close(self):
        self.persist()

if __name__ == "__main__":
    from services.research_service.data_processing.doc_processing.doc_processor import DocumentProcessor
    from services.research_service.embeddings.embedding_generator import EmbeddingGenerator

    print("📄 Processing document...")
    doc_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()

    try:
        # -------------------------------
        # Step 1: Process Document
        # -------------------------------
        chunks = doc_processor.process_document(r"C:\Users\kanis\FusionAI\services\research_service\vector_database\hydrogen_2.pdf")
        print(f"Chunks: {len(chunks)}")

        # -------------------------------
        # Step 2: Generate Embeddings
        # -------------------------------
        print("\n🧠 Generating embeddings...")
        embedded_chunks = embedding_generator.generate_embeddings(chunks)
        print(f"Embeddings: {len(embedded_chunks)}")

        # -------------------------------
        # Step 3: Debug Embedding
        # -------------------------------
        sample_embedding = embedded_chunks[0].embedding.tolist()

        print("\n🔍 Sample embedding check:")
        print("Type:", type(sample_embedding))
        print("Dimension:", len(sample_embedding))
        print("Preview:", sample_embedding[:5])

        # -------------------------------
        # Step 4: Initialize Chroma DB
        # -------------------------------
        print("\n🗄️ Initializing ChromaDB...")
        vector_db = ChromaVectorDatabase()

        # ⚠️ IMPORTANT: reset after dimension issues
        print("♻️ Resetting collection (for fresh schema)...")
        vector_db.reset_collection()

        # -------------------------------
        # Step 5: Insert Embeddings
        # -------------------------------
        print("\n📦 Inserting embeddings...")
        inserted_ids = vector_db.insert_embeddings(embedded_chunks)
        print(f"Inserted {len(inserted_ids)} embeddings")

        # -------------------------------
        # Step 6: Query Embedding
        # -------------------------------
        query_text = "What is the main topic of the document?"

        print("\n🔎 Generating query embedding...")
        query_vector = embedding_generator.generate_query_embedding(query_text)

        print("Query dimension:", len(query_vector))

        # -------------------------------
        # Step 7: Search
        # -------------------------------
        print("\n🔍 Searching...")
        results = vector_db.search(query_vector.tolist(), limit=5)

        # -------------------------------
        # Step 8: Display Results
        # -------------------------------
        print("\n📊 Search Results:\n")

        for i, result in enumerate(results):
            print(f"Result {i+1}")
            print(f"Score: {result['score']:.4f}")
            print(f"Content: {result['content'][:200]}...")
            print(f"Source: {result['citation']['source_file']}")
            print(f"Page: {result['citation']['page_number']}")
            print("-" * 50)

    except Exception as e:
        print(f"\n❌ PIPELINE ERROR: {e}")

    finally:
        if 'vector_db' in locals():
            vector_db.close()
            print("\n🔒 Database closed")