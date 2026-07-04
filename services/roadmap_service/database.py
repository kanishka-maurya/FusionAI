import os
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

def get_supabase(token: str) -> Client:
    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
        )


def insert_roadmap(client: Client, data: dict):
    client.table("roadmaps").insert(data).execute()


def get_roadmap(client: Client, roadmap_id: str):
    res =client.table("roadmaps") \
        .select("*") \
        .eq("roadmap_id", roadmap_id) \
        .execute()
    return res.data


def get_user_roadmaps(client: Client, user_id: str):
    res = client.table("roadmaps") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .execute()
    return res.data


def insert_nodes_bulk(client: Client, roadmap_id: str, nodes: list[dict]):
    rows = [
        {
            "node_id": node["id"],
            "roadmap_id": roadmap_id,
            "title": node["title"],
            "type": node["type"],
            "level": node["level"],
            "status": node["status"],
            "dependencies": node.get("dependencies", []),
            "position_x": node["position"]["x"],
            "position_y": node["position"]["y"],
            "content_generated": node["content_generated"]
        }
        for node in nodes
    ]

    client.table("nodes").insert(rows).execute()


def get_nodes_for_roadmap(client: Client, roadmap_id: str):
    res = client.table("nodes") \
        .select("*") \
        .eq("roadmap_id", roadmap_id) \
        .execute()

    return res.data


def get_node(client: Client, roadmap_id: str, node_id: str):
    res = client.table("nodes") \
        .select("*") \
        .eq("roadmap_id", roadmap_id) \
        .eq("node_id", node_id) \
        .single() \
        .execute()

    return res.data


def update_node_status(client: Client, roadmap_id: str, node_id: str, status: str):
    client.table("nodes") \
        .update({"status": status}) \
        .eq("roadmap_id", roadmap_id) \
        .eq("node_id", node_id) \
        .execute()


def mark_content_generated(client: Client, roadmap_id: str, node_id: str, content: dict):
    client.table("nodes") \
        .update({
            "content_generated": True,
            "raw_content": content
        }) \
        .eq("roadmap_id", roadmap_id) \
        .eq("node_id", node_id) \
        .execute()
