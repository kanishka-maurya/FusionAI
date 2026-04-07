from fastapi import APIRouter,HTTPException, Request
from services.research_service.data_processing.doc_processing.doc_processor import DocumentProcessor
from services.research_service.embeddings.embedding_generator import EmbeddingGenerator 
from services.research_service.vector_database.vector_database import ChromaVectorDatabase
from pydantic import BaseModel
router=APIRouter()

class TextContent(BaseModel):
   fileName:str
   copiedText:str

@router.post("/process")
async def process_text(text: TextContent,
                       request:Request):
    try:
      filename=text.fileName
      content=text.copiedText
      user_id = getattr(request.state, "user_id", None)
      session_id = getattr(request.state, "notebook_id", None)

      print("processing text content",filename,content)
      processor=DocumentProcessor()
      chunks=processor._create_chunks_from_text(content,filename,"text")
      embedding_generator=EmbeddingGenerator()
      embedded_chunks=embedding_generator.generate_embeddings(chunks)
      vector_db=ChromaVectorDatabase()
      inserted_ids=vector_db.insert_embeddings(embedded_chunks, user_id=user_id, session_id=session_id)
      return {
            "filename": filename,
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
       print(e)
       raise HTTPException(status_code=500,detail=str(e))