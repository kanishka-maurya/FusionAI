from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
import shutil
from backend.core.logging import logging
import uuid
import os

from services.research_service.data_processing.audio_processing.audio_transcriber import AudioTranscriber,transcribe_audio
from services.research_service.vector_database.vector_database import ChromaVectorDatabase
from services.research_service.embeddings.embedding_generator import EmbeddingGenerator
from services.research_service.generation.generation import RAGGenerator, RAGResult
    
    
router = APIRouter()
processor = AudioTranscriber()
embedding_generator = EmbeddingGenerator()
vector_db = ChromaVectorDatabase()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY environment variable not set")




@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        ext = Path(file.filename).suffix.lower()
        print("processing audio")
        print(file)
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        file_path = UPLOAD_DIR / unique_name
        print(file_path)
        if not file_path:
            print("you are going wrong")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        try:
          with open(file_path, "rb") as f:
            audio_bytes = f.read()

          results = processor.run_notebook_pipeline(audio_bytes)
        except Exception as e:
          print(e)
        print(results)
        file_path.unlink(missing_ok=True)
        try:
          embedded_chunks = embedding_generator.generate_embeddings(results["chunks"])
        except Exception as e:
          print(e)
        print(len(embedded_chunks))
        print(embedded_chunks)
        try:
          inserted_ids = vector_db.insert_embeddings(embedded_chunks)
        except Exception as e:
          print(e)
        print(f"Inserted {len(inserted_ids)} embeddings")
        return {
            "filename": file.filename,
            "total_chunks": len(results["chunks"]),
            "chunks_preview": [
                {
                    "content": chunk.content[:200],
                    "page": chunk.page_number,
                    "chunk_id": chunk.chunk_id
                }
                for chunk in results["chunks"][:5]
            ]
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    

