from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Query, Request
from backend.dependencies import get_vector_db

router = APIRouter()


@router.get("/get_contents")
async def get_notebook_sources( request: Request):
    user_id = getattr(request.state, "user_id", None)
    notebook_id = getattr(request.state, "notebook_id", None)
    print("something happeninggg",notebook_id)
    try:
        print("hereeeee")
        sources = get_vector_db().get_sources_by_session(
            user_id=user_id,
            session_id=notebook_id
        )
        print("sources related to this notebook",sources)
        return {
            "notebook_id": notebook_id,
            "sources": sources
        }

    except Exception as e:
        return {"error": str(e)}
@router.delete("/delete_contents")
async def delete_notebook_sources(request:Request):
    user_id = getattr(request.state, "user_id", None)
    notebook_id = getattr(request.state, "notebook_id", None)
    print("deleting contents for notebook",notebook_id)
    try:
        get_vector_db().delete_sources_by_session(
            user_id=user_id,
            session_id=notebook_id
        )
        return {
            "notebook_id": notebook_id,
            "message": "Sources deleted successfully"
        }
    except Exception as e:
        return {"error": str(e)}
    
@router.delete("/delete_source")
async def delete_single_source(
    request:Request,
    source_name: str = Query(...),
):
    """
    Delete a single source from a notebook (session)
    """
    user_id = getattr(request.state, "user_id", None)
    notebook_id = getattr(request.state, "notebook_id", None)
    if not user_id or not notebook_id:
        raise HTTPException(status_code=400, detail="Missing required headers")

    try:
        result = get_vector_db().delete_single_source(
            user_id=user_id,
            session_id=notebook_id,
            source_name=source_name,
        )

        deleted_count = result.get("deleted_count", 0)
        if deleted_count == 0:
            return {
                "success": False,
                "message": "No matching source found",
                "deleted_count": 0,
            }

        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} chunks of '{source_name}'",
        }

    except Exception as e:
        print("Delete source error:", e)
        raise HTTPException(status_code=500, detail="Failed to delete source")
