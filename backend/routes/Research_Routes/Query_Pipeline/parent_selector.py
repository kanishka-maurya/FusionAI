from backend.routes.Research_Routes.Nexus_Graph_DB.services import (
    engine_services
)
from backend.routes.Research_Routes.utils import hybrid_relevance_score


class ParentSelector:

    async def select_best_parent(self, routed_query):

        best_score = -1.0
        best_parent = None

        for root_id in routed_query["candidate_roots"]:

            node = await engine_services.get_node(root_id)

            if not node:
                continue

            score = engine_services.cosine_similarity(
                routed_query["embedding"],
                node["node_embedding"]
            )
            score_details = hybrid_relevance_score(
                routed_query["query"],
                routed_query["entities"],
                node,
                score
            )

            if score_details["hybrid_score"] > best_score:

                best_score = score_details["hybrid_score"]
                best_parent = node
                best_score_details = score_details
        return {
            "query": routed_query["query"],
            "entities": routed_query["entities"],
            "best_parent": best_parent,
            "score": best_score,
            "score_details": best_score_details if best_parent else {},
            "fallback_used": routed_query["fallback_used"]
        }


parent_selector = ParentSelector()
