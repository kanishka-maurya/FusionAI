from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.document import router as document_router
from backend.routes.youtube_video import router as youtube_router
from backend.routes.web import router as web_router
from backend.routes.audio import router as audio_router
from backend.routes.text_content import router as text_content_router

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(document_router, prefix="/api/documents", tags=["Documents"])
app.include_router(youtube_router, prefix="/api/youtube", tags=["Youtube Videos"])
app.include_router(web_router,prefix="/api/web",tags=["Web Pages"])
app.include_router(audio_router,prefix="/api/audio",tags=["Audio Files"])
app.include_router(text_content_router,prefix="/api/text",tags=["Text Content"])