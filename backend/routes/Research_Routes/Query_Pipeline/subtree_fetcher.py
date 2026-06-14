from backend.routes.Research_Routes.Nexus_Graph_DB.services import (
    engine_services
)
from backend.routes.Research_Routes.utils import build_provenance


class SubtreeFetcher:

    async def fetch(self, parent_payload):

        parent_node = parent_payload["best_parent"]

        if not parent_node:
            return None

        subtree = await engine_services.fetch_subtree(
            parent_node["node_id"]
        )

        return {
            "query": parent_payload["query"],
            "entities": parent_payload["entities"],
            "root_node": parent_node,
            "relevance_score": parent_payload["score"],
            "score_details": parent_payload.get("score_details", {}),
            "subtree": subtree,
            "provenance": [
                build_provenance(node)
                for node in subtree
            ],
            "fallback_used": parent_payload["fallback_used"]
        }


subtree_fetcher = SubtreeFetcher()
