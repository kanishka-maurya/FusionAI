from fastapi import APIRouter, HTTPException, Request,Header
from backend.dependencies import get_embedding_generator, get_vector_db
from services.research_service.data_processing.audio_processing.youtube_transcriber import YoutubeTranscriber
router=APIRouter()

transcriber=YoutubeTranscriber()
@router.post("/process_video_link")
async def process_video_link(video_link:str,
                              request:Request):
    try:
        print("coming to youtube route")
        video_id = transcriber._extract_video_id(video_link)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        chunks = transcriber.process_transcript(url=video_link, video_id=video_id)
        if not chunks:
            raise HTTPException(status_code=422, detail="No transcript text was found for this video")
        user_id = getattr(request.state, "user_id", None)
        session_id = getattr(request.state, "notebook_id", None)
        if not user_id or not session_id:
            raise HTTPException(status_code=400, detail="Missing user or notebook context")
        """ print(f"Transcribed {len(chunks)} utterances:")
        for chunk in chunks[:5]:
            print(f"  {chunk.content}")"""
        print(chunks)
        print("generating embeddings")
        embedded_chunks = get_embedding_generator().generate_embeddings(chunks)
        print(len(embedded_chunks))
        print("now here")
        inserted_ids = get_vector_db().insert_embeddings(embedded_chunks,  user_id=user_id, session_id=session_id)
        print(f"Inserted {inserted_ids} embeddings")
        return {
            "filename": video_link,
            "url": video_link,
            "video_id": video_id,
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
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"YouTube processing failed: {str(e)}")
