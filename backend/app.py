from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.document import router as document_router

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)
app.include_router(document_router, prefix="/api/documents", tags=["Documents"])