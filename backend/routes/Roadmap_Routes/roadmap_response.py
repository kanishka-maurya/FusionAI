import uuid
import json
from fastapi import APIRouter, HTTPException,Request
from services.roadmap_service.data_models import GenerateRoadmapRequest, RoadmapResponse
import services.roadmap_service.database as db
from services.roadmap_service.llm_service import generate_roadmap_json, generate_node_content

router = APIRouter()


def normalize_node_content(content, node_title="", roadmap_title=""):

    if isinstance(content, str):

        try:
            content = json.loads(content)
        except Exception:
            content = {
                "summary": content
            }

    if not isinstance(content, dict):
        content = {}

    topics = content.get("topics")
    if not isinstance(topics, list):
        topics = []

    normalized_topics = []
    for index, topic in enumerate(topics[:5], start=1):
        if isinstance(topic, dict):
            normalized_topics.append({
                "title": topic.get("title") or f"Concept {index}",
                "explanation": topic.get("explanation") or topic.get("summary") or "",
                "code_example": topic.get("code_example"),
                "key_takeaway": topic.get("key_takeaway") or ""
            })
        elif topic:
            normalized_topics.append({
                "title": f"Concept {index}",
                "explanation": str(topic),
                "code_example": None,
                "key_takeaway": ""
            })

    if not normalized_topics:
        normalized_topics = [{
            "title": node_title or "Topic overview",
            "explanation": (
                content.get("summary")
                or f"This section introduces {node_title or 'the selected topic'} "
                f"within {roadmap_title or 'the roadmap'}."
            ),
            "code_example": None,
            "key_takeaway": (
                f"Understand the role of {node_title or 'this topic'} before "
                "moving to dependent roadmap nodes."
            )
        }]

    def list_or_default(value, fallback):
        if isinstance(value, list):
            return [str(item) for item in value if item]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return fallback

    resources = content.get("resources")
    if not isinstance(resources, list):
        resources = []

    normalized_resources = []
    for resource in resources[:5]:
        if isinstance(resource, dict):
            normalized_resources.append({
                "type": resource.get("type") or "article",
                "title": resource.get("title") or "Learning resource",
                "url": resource.get("url") or "#"
            })

    if not normalized_resources:
        normalized_resources = [{
            "type": "article",
            "title": f"Search: {node_title}",
            "url": "#"
        }]

    practice_questions = content.get("practice_questions")
    if not isinstance(practice_questions, list):
        practice_questions = []

    normalized_questions = []
    for index, question in enumerate(practice_questions[:4], start=1):
        if isinstance(question, dict):
            normalized_questions.append({
                "question": question.get("question") or f"Practice question {index}",
                "hint": question.get("hint") or "",
                "answer": question.get("answer") or ""
            })
        elif question:
            normalized_questions.append({
                "question": str(question),
                "hint": "",
                "answer": ""
            })

    if not normalized_questions:
        normalized_questions = [{
            "question": f"How would you explain {node_title or 'this topic'} in your own words?",
            "hint": "Focus on the goal, core idea, and one concrete example.",
            "answer": ""
        }]

    return {
        "summary": (
            content.get("summary")
            or f"Learn the core ideas behind {node_title or 'this roadmap topic'}."
        ),
        "estimated_time": content.get("estimated_time") or "1 week",
        "what_you_will_learn": list_or_default(
            content.get("what_you_will_learn"),
            [
                f"Understand {node_title or 'the topic'} conceptually",
                "Identify where it fits in the roadmap",
                "Apply the idea through practice"
            ]
        ),
        "topics": normalized_topics,
        "common_misconceptions": list_or_default(
            content.get("common_misconceptions"),
            []
        ),
        "resources": normalized_resources,
        "practice_questions": normalized_questions
    }

@router.post("/generate", response_model=RoadmapResponse)
async def generate_roadmap(request: GenerateRoadmapRequest, req: Request):

    roadmap_id = str(uuid.uuid4())

    try:
        raw =await generate_roadmap_json(request.topic, request.level.value)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=502,
                            detail=f"Roadmap generation failed: {str(e)}")
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


@router.get("/{roadmap_id}")
async def get_roadmap(roadmap_id: str, req: Request):
    client = db.get_supabase(req.state.token)
    client.auth.set_session(req.state.token, req.state.token)

    roadmaps = db.get_roadmap(client, roadmap_id)
    if not roadmaps:
        raise HTTPException(status_code=404, detail="Roadmap not found")

    roadmap = roadmaps[0]
    if roadmap.get("user_id") != req.state.user_id:
        raise HTTPException(status_code=403, detail="Roadmap does not belong to this user")

    nodes = db.get_nodes_for_roadmap(client, roadmap_id)
    shaped_nodes = [
        {
            "node_id": n["node_id"],
            "roadmap_id": n["roadmap_id"],
            "title": n["title"],
            "type": n["type"],
            "level": n["level"],
            "status": n["status"],
            "dependencies": n.get("dependencies") or [],
            "position_x": n.get("position_x"),
            "position_y": n.get("position_y"),
            "content_generated": n.get("content_generated"),
            "raw_content": n.get("raw_content"),
            "created_at": n.get("created_at"),
        }
        for n in nodes
    ]

    return {
        **roadmap,
        "nodes": shaped_nodes,
        "total_nodes": roadmap.get("total_nodes") or len(shaped_nodes),
    }


@router.get("/{roadmap_id}/node/{node_id}/content")
async def get_node_content(roadmap_id: str, node_id: str, req: Request):
    client = db.get_supabase(req.state.token)
    client.auth.set_session(req.state.token, req.state.token)

    roadmaps = db.get_roadmap(client, roadmap_id)
    if not roadmaps:
        raise HTTPException(status_code=404, detail="Roadmap not found")

    roadmap = roadmaps[0]
    if roadmap.get("user_id") != req.state.user_id:
        raise HTTPException(status_code=403, detail="Roadmap does not belong to this user")

    node = db.get_node(client, roadmap_id, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    if node.get("content_generated") and node.get("raw_content"):
        content = normalize_node_content(
            node["raw_content"],
            node.get("title", ""),
            roadmap.get("title", "")
        )
        return {"source": "db", "content": content}

    dep_ids = node.get("dependencies") or []
    prerequisites = []
    for dep_id in dep_ids:
        dep_node = db.get_node(client, roadmap_id, dep_id)
        if dep_node:
            prerequisites.append(dep_node["title"])

    try:
        content = await generate_node_content(
            node_title=node["title"],
            roadmap_title=roadmap["title"],
            level=node["level"],
            prerequisites=prerequisites,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Content generation failed: {str(e)}")

    content = normalize_node_content(
        content,
        node.get("title", ""),
        roadmap.get("title", "")
    )

    db.mark_content_generated(client, roadmap_id, node_id, content)
    return {"source": "generated", "content": content}


@router.patch("/{roadmap_id}/node/{node_id}/status")
async def update_node_status(roadmap_id: str, node_id: str, status: str, req: Request):
    valid_statuses = {"done", "completed", "in_progress", "skipped", "unlocked"}
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")

    client = db.get_supabase(req.state.token)
    client.auth.set_session(req.state.token, req.state.token)

    roadmaps = db.get_roadmap(client, roadmap_id)
    if not roadmaps:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    if roadmaps[0].get("user_id") != req.state.user_id:
        raise HTTPException(status_code=403, detail="Roadmap does not belong to this user")

    normalized_status = "done" if status == "completed" else status
    db.update_node_status(client, roadmap_id, node_id, normalized_status)

    if normalized_status == "done":
        nodes = db.get_nodes_for_roadmap(client, roadmap_id)
        node_status_map = {n["node_id"]: n["status"] for n in nodes}
        for node in nodes:
            if node["status"] != "locked":
                continue
            deps = node.get("dependencies") or []
            if deps and all(node_status_map.get(dep) in {"done", "completed"} for dep in deps):
                db.update_node_status(client, roadmap_id, node["node_id"], "unlocked")

    return {"message": f"Node {node_id} updated to '{normalized_status}'"}

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
