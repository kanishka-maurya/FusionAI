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
        collection_name: str="FusionAI_Collection"
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        logger.info(f"ChromaDB initialized at {self.db_path}")

    def insert_embeddings(self, embedded_chunks: List[EmbeddedChunk], user_id:str , session_id: str) -> List[str]:
        if not embedded_chunks:
            return []
        
        try:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for chunk in embedded_chunks:
                data = chunk.to_vector_db_format(user_id=user_id, session_id=session_id)

                ids.append(data["id"])
                documents.append(data["content"])
                embeddings.append(data["vector"])
                
                print("we are inside insert_embedding func....")
                print(user_id)
                print(session_id)
                
                metadata = {
                    "user_id": str(user_id),
                    "session_id": str(session_id),
                    "source_file": data.get("source_file"),
                    "source_type": data.get("source_type"),
                    "page_number": data.get("page_number", -1),
                    "chunk_index": data.get("chunk_index"),
                    "start_char": data.get("start_char", -1),
                    "end_char": data.get("end_char", -1),
                    "embedding_model": data.get("embedding_model"),
                }

        
                if isinstance(data.get("metadata"), dict):
                   for key, value in data["metadata"].items():
                      metadata[key] = value
                metadata["user_id"] = user_id
                metadata["session_id"] = session_id
    

                clean_metadata = {}
                for key, value in metadata.items():
                  if value is None:
                    continue
                  elif isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                  else:
                    clean_metadata[key] = str(value)
                cnt=0
                for key, value in clean_metadata.items():
                   if key=="user_id" or key=="session_id":
                       print(f"{key}:{value}")
                       cnt+=1
                   elif value is None:
                       print(f"this is none: {key}")
                   else:
                       print(f"{key}")
                logging.info(clean_metadata)
                metadatas.append(clean_metadata)
            print("metadatas appended",metadatas)
            try:
             self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            except Exception as e:
                print(e)
            logging.info(f"Inserted {len(ids)} embeddings")
            return ids

        except Exception as e:
                error = CustomException(e, sys)
                logging.error(error)
                raise error

    def search(
        self,
        query_vector,
        limit,
        user_id,
        session_id
    ) :

        try:
            where = None

            if user_id and session_id:
                print("we are inside search func of vector DB....")
                print(user_id)
                print(session_id)
                where = {
                    "$and": [
                        {"user_id": str(user_id)},
                        {"session_id": str(session_id)}
                    ]
                }
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where= where
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

            all_data = self.collection.get()
            logging.info(all_data["metadatas"][:5])
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
            
    def get_sources_by_session(self, user_id: str, session_id: str):
      try:
        # Fetch for the user
        results = self.collection.get(where={"user_id": str(user_id)})
        metadatas = results.get("metadatas", [])
        unique_sources = {}

        for metadata in metadatas:
            # 1. Get the session ID from metadata
            nb_id = metadata.get("session_id")
            s_type = metadata.get("source_type") or "text"
            s_file = metadata.get("source_file")
            source_url = (
                metadata.get("source_url")
                or metadata.get("video_url")
                or metadata.get("original_url")
                or metadata.get("url_fragment")
            )
            if source_url and "#chunk-" in str(source_url):
                source_url = str(source_url).split("#chunk-")[0]
            # 2. Safety Check: Convert both to strings to avoid mismatch
            if str(nb_id) != str(session_id):
                continue

            display_type = self._map_source_type(s_type)
            display_name = s_file

            if s_type == "youtube":
                display_name = source_url or s_file or "YouTube Video"
            elif s_type == "web":
                display_name = (
                    metadata.get("title")
                    or s_file
                    or source_url
                    or "Web Page"
                )
            elif s_type == "audio":
                headline = metadata.get("headline")
                if headline:
                    display_name = f"Audio: {headline[:48]}..."
                elif not s_file or s_file == "audio_source":
                    display_name = "Audio Recording"
            elif s_type in {"pdf", "txt", "text"}:
                display_name = s_file or metadata.get("title")
            
            if not display_name:
                display_name = "Untitled Source"

            if display_name not in unique_sources:
                unique_sources[display_name] = {
                    "id": display_name, 
                    "name": display_name,
                    "type": display_type,
                    "rawType": s_type,
                    "url": source_url,
                    "pages": metadata.get("total_pages")
                }

        print(f"Final sources list: {list(unique_sources.values())}")
        return list(unique_sources.values())

      except Exception as e:
        print(f"Error in get_sources: {e}")
        return []
      
    def delete_sources_by_session(self, user_id: str, session_id: str):
      try:
        results = self.collection.get(where={"user_id": str(user_id)})

        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])

        ids_to_delete = []

        for doc_id, metadata in zip(ids, metadatas):
            nb_id = metadata.get("session_id")
            if str(nb_id) == str(session_id):
                ids_to_delete.append(doc_id)
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} sources for session {session_id}")
        else:
            print("No sources found to delete")

        return {
            "success": True,
            "deleted_count": len(ids_to_delete)
        }

      except Exception as e:
        print(f"Error in delete_sources_by_session: {e}")
        return {
            "success": False,
            "error": str(e)
        }
      
    def delete_single_source(self, user_id: str, session_id: str, source_name: str):
     try:
        results = self.collection.get(where={"user_id": str(user_id)})

        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])

        ids_to_delete = []

        for doc_id, metadata in zip(ids, metadatas):
            source_values = {
                str(metadata.get("source_file") or ""),
                str(metadata.get("source_url") or ""),
                str(metadata.get("video_url") or ""),
                str(metadata.get("original_url") or ""),
                str(metadata.get("url_fragment") or "").split("#chunk-")[0],
            }
            if (
                str(metadata.get("session_id")) == str(session_id)
                and source_name in source_values
            ):
                ids_to_delete.append(doc_id)

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        return {"deleted_count": len(ids_to_delete)}

     except Exception as e:
        return {"error": str(e)}
    def _map_source_type(self, source_type: str):
      if source_type == "pdf":
        return "PDF Document"
      elif source_type == "youtube":
        return "YouTube"
      elif source_type == "audio":
        return "Audio File"
      elif source_type == "web":
        return "Web URL"
      elif source_type in {"text", "txt", "md"}:
        return "Copied Text"
      return "Unknown"

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
