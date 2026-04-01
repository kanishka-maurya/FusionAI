from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
import shutil
import uuid

from services.research_service.data_processing.doc_processing.doc_processor import DocumentProcessor
from services.research_service.vector_database.vector_database import ChromaVectorDatabase
from services.research_service.embeddings.embedding_generator import EmbeddingGenerator
    
    
router = APIRouter()
processor = DocumentProcessor()
embedding_generator = EmbeddingGenerator()
vector_db = ChromaVectorDatabase()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        ext = Path(file.filename).suffix.lower()
        if ext not in processor.supported_formats:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        file_path = UPLOAD_DIR / unique_name

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        chunks = processor.process_document(file_path)
        file_path.unlink(missing_ok=True)
        embedded_chunks = embedding_generator.generate_embeddings(chunks)
        print(len(embedded_chunks))
        print(embedded_chunks)

        inserted_ids = vector_db.insert_embeddings(embedded_chunks)
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
async def query_documents(q:str):
    try:
        print(q)
        query_vector=embedding_generator.generate_query_embedding(q)
        results=vector_db.search(query_vector.tolist(),limit=5)
        print(results)
        return {"results":results}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
