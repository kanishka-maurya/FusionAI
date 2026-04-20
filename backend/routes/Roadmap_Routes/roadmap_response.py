import uuid
import json
from fastapi import APIRouter, HTTPException,Request
from services.roadmap_service.data_models import GenerateRoadmapRequest, RoadmapResponse
import services.roadmap_service.database as db
import cache
from services.roadmap_service.llm_service import generate_roadmap_json, generate_node_content

router = APIRouter()

@router.post("/generate", response_model=RoadmapResponse)
async def generate_roadmap(request: GenerateRoadmapRequest, req: Request):

    roadmap_id = str(uuid.uuid4())

    try:
        raw =await generate_roadmap_json(request.topic, request.level.value)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=502,
                            detail=f"Gemini generation failed: {str(e)}")
    print(raw)
    nodes = raw.get("nodes", [])
    if not nodes:
        raise HTTPException(status_code=500,
                            detail="Gemini returned an empty roadmap")

    for node in nodes:
        node["status"] = "unlocked" if not node.get("dependencies") else "locked"
        node["content_generated"] = False

    client = db.get_supabase(req.state.token)
    client.auth.set_session(req.state.token, req.state.token)
    db.insert_roadmap(client, {
        "roadmap_id": roadmap_id,
        "user_id": req.state.user_id,
        "title": raw["title"],
        "topic": request.topic,
        "description": raw.get("description", ""),
        "total_nodes": len(nodes),
    })
    print(nodes)
    db.insert_nodes_bulk(client, roadmap_id, nodes)

    response_data = {
        "roadmap_id": roadmap_id,
        "user_id": req.state.user_id,
        "title": raw["title"],
        "topic": request.topic,
        "description": raw.get("description", ""),
        "total_nodes": len(nodes),
        "nodes": nodes,
    }

    return response_data

@router.get("/user")
async def get_user_roadmaps(req: Request):
    client = db.get_supabase(req.state.token)
    client.auth.set_session(req.state.token, req.state.token)
    roadmaps = db.get_user_roadmaps(client, req.state.user_id)
    return {"roadmaps": roadmaps}

# # ─────────────────────────────────────────────────────────────────────────────
# # GET /api/roadmap/{roadmap_id}
# # Fetch a full roadmap — Supabase Redis → Supabase PostgreSQL fallback
# # ─────────────────────────────────────────────────────────────────────────────

# @router.get("/{roadmap_id}")
# async def get_roadmap(roadmap_id: str):
#     """
#     Cache-first fetch:
#     1. Check Supabase Redis — instant return if hit
#     2. Rebuild from Supabase PostgreSQL, repopulate cache
#     """

#     # Layer 1: Supabase Redis
#     cached = await cache.get_cached_roadmap(roadmap_id)
#     if cached:
#         return {**cached, "source": "cache"}

#     # Layer 2: Supabase PostgreSQL
#     roadmap = await db.get_roadmap(roadmap_id)
#     if not roadmap:
#         raise HTTPException(status_code=404, detail="Roadmap not found")

#     nodes = await db.get_nodes_for_roadmap(roadmap_id)

#     shaped_nodes = [
#         {
#             "id":                n["node_id"],
#             "title":             n["title"],
#             "type":              n["type"],
#             "level":             n["level"],
#             "status":            n["status"],
#             "dependencies":      n["dependencies"] or [],
#             "position":          {"x": n["position_x"], "y": n["position_y"]},
#             "content_generated": n["content_generated"],
#         }
#         for n in nodes
#     ]

#     response_data = {
#         "roadmap_id":  roadmap_id,
#         "user_id":     roadmap["user_id"],
#         "title":       roadmap["title"],
#         "topic":       roadmap["topic"],
#         "description": roadmap["description"],
#         "total_nodes": roadmap["total_nodes"],
#         "nodes":       shaped_nodes,
#     }

#     # Repopulate cache
#     await cache.cache_roadmap(roadmap_id, response_data)

#     return {**response_data, "source": "db"}


# # ─────────────────────────────────────────────────────────────────────────────
# # GET /api/roadmap/{roadmap_id}/node/{node_id}/content
# # Lazy content generation — generate only when user first clicks a node
# # ─────────────────────────────────────────────────────────────────────────────

# @router.get("/{roadmap_id}/node/{node_id}/content")
# async def get_node_content(roadmap_id: str, node_id: str):
#     """
#     3-layer content fetch:
#     1. Supabase Redis   → instant if previously generated
#     2. Supabase DB JSONB → generated before, Redis just expired
#     3. Gemini           → first time; save to DB + cache
#     """

#     # Layer 1: Supabase Redis
#     cached_content = await cache.get_cached_node_content(roadmap_id, node_id)
#     if cached_content:
#         return {"source": "cache", "content": cached_content}

#     # Layer 2: Supabase PostgreSQL (raw_content JSONB column)
#     node = await db.get_node(roadmap_id, node_id)
#     if not node:
#         raise HTTPException(status_code=404, detail="Node not found")

#     if node["content_generated"] and node.get("raw_content"):
#         content = node["raw_content"]
#         # raw_content from supabase-py comes back as a dict already (JSONB)
#         if isinstance(content, str):
#             content = json.loads(content)
#         await cache.cache_node_content(roadmap_id, node_id, content)
#         return {"source": "db", "content": content}

#     # Layer 3: Generate via Gemini
#     roadmap = await db.get_roadmap(roadmap_id)
#     if not roadmap:
#         raise HTTPException(status_code=404, detail="Roadmap not found")

#     # Resolve prerequisite node titles
#     dep_ids       = node.get("dependencies") or []
#     prerequisites = []
#     for dep_id in dep_ids:
#         dep_node = await db.get_node(roadmap_id, dep_id)
#         if dep_node:
#             prerequisites.append(dep_node["title"])

#     try:
#         content = await generate_node_content(
#             node_title    = node["title"],
#             roadmap_title = roadmap["title"],
#             level         = node["level"],
#             prerequisites = prerequisites,
#         )
#     except Exception as e:
#         raise HTTPException(status_code=502,
#                             detail=f"Content generation failed: {str(e)}")

#     # Persist to Supabase DB + cache in Supabase Redis
#     await db.mark_content_generated(roadmap_id, node_id, content)
#     await cache.cache_node_content(roadmap_id, node_id, content)

#     return {"source": "generated", "content": content}


# # ─────────────────────────────────────────────────────────────────────────────
# # PATCH /api/roadmap/{roadmap_id}/node/{node_id}/status
# # Update node progress + auto-unlock dependents
# # ─────────────────────────────────────────────────────────────────────────────

# @router.patch("/{roadmap_id}/node/{node_id}/status")
# async def update_node_status(roadmap_id: str, node_id: str, status: str):
#     """
#     1. Update status in Supabase DB
#     2. Auto-unlock nodes whose all dependencies are now done
#     3. Invalidate roadmap cache so frontend sees fresh statuses
#     """
#     valid_statuses = {"done", "in_progress", "skipped", "unlocked"}
#     if status not in valid_statuses:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid status. Must be one of: {valid_statuses}"
#         )

#     await db.update_node_status(roadmap_id, node_id, status)

#     # Auto-unlock dependents when a node is marked done
#     if status == "done":
#         all_nodes = await db.get_nodes_for_roadmap(roadmap_id)
#         node_status_map = {n["node_id"]: n["status"] for n in all_nodes}

#         for n in all_nodes:
#             if n["status"] != "locked":
#                 continue
#             deps = n["dependencies"] or []
#             if not deps:
#                 continue
#             # Unlock if ALL dependencies are now done
#             if all(node_status_map.get(dep) == "done" for dep in deps):
#                 await db.update_node_status(roadmap_id, n["node_id"], "unlocked")

#     # Invalidate roadmap cache — next GET will rebuild from DB
#     await cache.invalidate_roadmap(roadmap_id)

#     return {"message": f"Node {node_id} updated to '{status}'"}


# # ─────────────────────────────────────────────────────────────────────────────
# # DELETE /api/roadmap/{roadmap_id}
# # Delete a roadmap and all its cached data
# # ─────────────────────────────────────────────────────────────────────────────

# @router.delete("/{roadmap_id}")
# async def delete_roadmap(roadmap_id: str):
#     """
#     Deletes roadmap from Supabase DB (cascades to nodes via FK)
#     and clears all related Redis cache keys at once.
#     """
#     client_sb = await db.get_supabase()

#     res = (
#         await client_sb.table("roadmaps")
#         .delete()
#         .eq("roadmap_id", roadmap_id)
#         .execute()
#     )

#     if not res.data:
#         raise HTTPException(status_code=404, detail="Roadmap not found")

#     # Wipe all cache keys for this roadmap in one SCAN+DEL pass
#     await cache.invalidate_all_roadmap_content(roadmap_id)

#     return {"message": f"Roadmap {roadmap_id} deleted"}