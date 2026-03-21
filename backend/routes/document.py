from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
import shutil
import uuid

from services.research_service.data_processing.doc_processing.doc_processor import DocumentProcessor

router = APIRouter()
processor = DocumentProcessor()

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