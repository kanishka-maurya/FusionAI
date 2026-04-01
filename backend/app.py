from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.document import router as document_router
from backend.routes.youtube_video import router as youtube_router
app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)
app.include_router(document_router, prefix="/api/documents", tags=["Documents"])
app.include_router(youtube_router, prefix="/api/youtube", tags=["Youtube Videos"])
