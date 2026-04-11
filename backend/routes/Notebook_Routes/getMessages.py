from fastapi import APIRouter, Header, HTTPException
from typing import Optional, List, Dict, Any
import os

from zep_cloud.client import Zep

router = APIRouter()

zep_client = Zep(api_key=os.getenv("ZEP_API_KEY"))


@router.get("/chat/messages")
def get_chat_messages(
    x_notebook_id: Optional[str] = Header(None),
):
    """
    Fetch full chat history for a notebook (thread)

    Header:
        X-Notebook-Id: notebook_id (used as session_id)

    Returns:
        List of messages in chronological order
    """

    if not x_notebook_id:
        raise HTTPException(status_code=400, detail="Missing X-Notebook-Id header")

    try:
        try:
             response = zep_client.thread.get(thread_id=x_notebook_id)
        except Exception as e:
             print(e)

        messages = response.messages or []

        formatted_messages: List[Dict[str, Any]] = [
            {
                "id": str(i),
                "role": msg.role,  
                "content": msg.content,
                "timestamp": msg.created_at,
            }
            for i, msg in enumerate(messages)
        ]

        return {
            "success": True,
            "count": len(formatted_messages),
            "messages": formatted_messages,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reset_chat")
async def reset_chat(
    x_user_id: Optional[str] = Header(None),
    x_notebook_id: Optional[str] = Header(None),
):
    if not x_user_id or not x_notebook_id:
        raise HTTPException(status_code=400, detail="Missing headers")

    try:
        zep = Zep(api_key=os.getenv("ZEP_API_KEY"))

        zep.thread.delete(x_notebook_id)
        zep.thread.create(
            thread_id=x_notebook_id,
            user_id=x_user_id
        )

        return {"success": True, "message": "Chat reset successfully"}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Failed to reset chat")