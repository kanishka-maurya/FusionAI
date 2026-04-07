from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Header
from typing import List
from pathlib import Path
import shutil
from backend.core.logging import logging
import uuid
import os
from pydantic import BaseModel


from services.research_service.data_processing.doc_processing.doc_processor import DocumentProcessor
from services.research_service.vector_database.vector_database import ChromaVectorDatabase
from services.research_service.embeddings.embedding_generator import EmbeddingGenerator
from services.research_service.generation.generation import RAGGenerator
from memory.memory import NotebookMemoryLayer
    
    
router = APIRouter()
processor = DocumentProcessor()
embedding_generator = EmbeddingGenerator()
vector_db = ChromaVectorDatabase()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

llm_api_key = os.getenv("GROQ_API_KEY")
if not llm_api_key:
    raise RuntimeError("GROQ_API_KEY environment variable not set")

memory_api_key = os.getenv("ZEP_API_KEY")
if not memory_api_key:
    raise RuntimeError("ZEP_API_KEY environment variable not set")



@router.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    
    user_id = getattr(request.state, "user_id", None)
    notebook_id = getattr(request.state, "notebook_id", None)

    try:
        ext = Path(file.filename).suffix.lower()
        print("processing doc")
        print(file.filename)
        if ext not in processor.supported_formats:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        file_path = UPLOAD_DIR / unique_name
        print(file_path)
        if not file_path:
            print("you are going wrong")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        try:
          chunks = processor.process_document(file_path)
        except Exception as e:
          print(e)
        print(chunks)
        file_path.unlink(missing_ok=True)
        try:
          embedded_chunks = embedding_generator.generate_embeddings(chunks)
        except Exception as e:
          print(e)
        print(len(embedded_chunks))
        print(embedded_chunks)
        try:
          inserted_ids = vector_db.insert_embeddings(embedded_chunks,  user_id=user_id, session_id=notebook_id)
        except Exception as e:
          print(e)
        print(f"Inserted {len(inserted_ids)} embeddings")
        return {
            "filename": file.filename,
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
    

@router.get("/query")
async def query_documents(q: str, request:Request):
    try:
        user_id = getattr(request.state, "user_id", None)
        notebook_id = getattr(request.state, "notebook_id", None)

        print("user_id from middleware:", user_id)
        print("notebook_id from middleware:", notebook_id)
        print(q)
        if not q or not q.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        rag_generator = RAGGenerator(
         embedding_generator=embedding_generator,
         vector_db=vector_db,
         api_key=llm_api_key,
         temperature=0.1
        )
        

        memory_layer = NotebookMemoryLayer(
           user_id = user_id,
           session_id = notebook_id,
           zep_api_key= memory_api_key
        )

        memory = memory_layer.build_memory_context(q)
        
        result = rag_generator.generate_results(
            query=q,
            memory= memory,
            user_id=user_id,
            session_id=notebook_id

        )

        memory_layer.save_conversation_turn(result)
        print("results to be sent",result.response)
        return {
            "results": result.response,
            "user_id": user_id,
            "session_id": notebook_id
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500,detail=str(e))
