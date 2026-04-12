import uuid
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
from memory.memory import NotebookMemoryLayer
import os

zep_api_key=os.getenv("ZEP_API_KEY")
router=APIRouter()
class NotebookCreate(BaseModel):
    name: str
    description: Optional[str]=None

@router.post("/create")
async def create_notebook(notebook:NotebookCreate,request:Request):
      user_id = request.state.user_id 
      notebook_id = f"nb_{uuid.uuid4().hex[:12]}" 
      notebook_memory=NotebookMemoryLayer(user_id=user_id,session_id=notebook_id,zep_api_key=zep_api_key,create_new_session=True)
      notebook_memory._setup_user_and_session(create_new_session=True)
      return {
            "notebook_id": notebook_id,
            "name": notebook.name,
            "description": notebook.description,
            "message": "Notebook space initialized"
    }