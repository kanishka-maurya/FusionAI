from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from backend.routes.document import router as document_router
from backend.routes.youtube_video import router as youtube_router
from backend.routes.web import router as web_router
from backend.routes.audio import router as audio_router
from backend.routes.text_content import router as text_content_router
from backend.routes.Notebook_Routes.getContents import router as get_notebook_contents_router
from backend.routes.Notebook_Routes.getMessages import router as chat_router
import httpx
import os
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from backend.routes.Roadmap_Routes.roadmap_response import router as roadmap_router
from backend.routes.Research_Routes.get_Latest_Data import router as research_router
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in ["/docs", "/openapi.json", "/health"]:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        notebook_id = request.headers.get("X-Notebook-Id")
        print(
            "auth header present",
            bool(auth_header)
        )
        print(
            "notebook id present",
            bool(notebook_id)
        )
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing token"})

        token = auth_header.split(" ")[1]
        request.state.token = token
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{SUPABASE_URL}/auth/v1/user",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": SUPABASE_ANON_KEY
                    }
                )
                print(response)
            
            if response.status_code == 200:
                user_data = response.json()
                
                request.state.user_id = user_data.get("id")
                request.state.notebook_id = notebook_id 
                
                return await call_next(request)
            else:
                return JSONResponse(status_code=401, content={"detail": "Invalid token from Supabase"})

        except Exception as e:
            print(f"Auth error: {str(e)}")
            return JSONResponse(status_code=401, content={"detail": f"Auth error: {str(e)}"})

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)
app.include_router(document_router, prefix="/api/documents", tags=["Documents"])
app.include_router(youtube_router, prefix="/api/youtube", tags=["Youtube Videos"])
app.include_router(web_router,prefix="/api/web",tags=["Web Pages"])
app.include_router(audio_router,prefix="/api/audio",tags=["Audio Files"])
app.include_router(text_content_router,prefix="/api/text",tags=["Text Content"])
app.include_router(get_notebook_contents_router,prefix="/api/notebooks",tags=["Notebooks"])
app.include_router(chat_router,prefix="/api/notebooks",tags=["Notebook Chats"])
app.include_router(roadmap_router,prefix="/api/roadmap",tags=["Roadmap"])
app.include_router(research_router,prefix="/ai-news",tags=["AI News"])
