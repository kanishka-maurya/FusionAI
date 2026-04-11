from fastapi import APIRouter, UploadFile, File, HTTPException, Header,Request
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
async def upload_document(request:Request,file: UploadFile = File(...)):
    try:
        user_id = getattr(request.state, "user_id", None)
        session_id = getattr(request.state, "notebook_id", None)
        ext = Path(file.filename).suffix.lower()
        print("processing audio")
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        file_path = UPLOAD_DIR / unique_name
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
            logging.error(f"Transcription failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to transcribe audio")
        file_path.unlink(missing_ok=True)
        try:
          embedded_chunks = embedding_generator.generate_embeddings(results["chunks"])
        except Exception as e:
          print(e)
        for chunk in embedded_chunks:
            if not hasattr(chunk, "metadata") or chunk.metadata is None:
                chunk.metadata = {}
            
            chunk.metadata.update({
                "source_file": file.filename,
                "source_type": "audio",
                "user_id": user_id,
                "session_id": session_id 
            })

        print(f"DEBUG: Session ID being sent to DB for audio: {session_id}")
        try:
          print("session id coming:",session_id)
          print("embedded chunks going to vector_db",embedded_chunks)
          inserted_ids = vector_db.insert_embeddings(embedded_chunks,  user_id=user_id, session_id=session_id)
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
    

