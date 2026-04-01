from fastapi import APIRouter 
from services.research_service.data_processing.audio_processing.youtube_transcriber import YoutubeTranscriber
from services.research_service.embeddings.embedding_generator import EmbeddingGenerator
from services.research_service.vector_database.vector_database import ChromaVectorDatabase
router=APIRouter()

transcriber=YoutubeTranscriber()
embedding_generator=EmbeddingGenerator()
vector_db=ChromaVectorDatabase()
@router.post("/process_video_link")
async def process_video_link(video_link:str):
    try:
        print("coming to youtube route")
        video_id = transcriber._extract_video_id(video_link)
        chunks = transcriber.process_transcript(url=video_link, video_id=video_id)
        
        """ print(f"Transcribed {len(chunks)} utterances:")
        for chunk in chunks[:5]:
            print(f"  {chunk.content}")"""
        print(chunks)
        print("generating embeddings")
        embedded_chunks = embedding_generator.generate_embeddings(chunks)
        print(len(embedded_chunks))
        print("now here")
        inserted_ids = vector_db.insert_embeddings(embedded_chunks)
        print(f"Inserted {len(inserted_ids)} embeddings")
        return {
            "filename": video_link,
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