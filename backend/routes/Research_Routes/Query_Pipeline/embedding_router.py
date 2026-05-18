from typing import Dict, List

from backend.routes.Research_Routes.Nexus_Graph_DB.services import (
    engine_services
)

from backend.routes.Research_Routes.Nexus_Graph_DB.config import (
    embedding_service
)


class EmbeddingRouter:

    async def route_queries(
        self,
        queries: List[str],
        entities_map: Dict[str, List[str]]
    ):

        routed_payload = []

        for query in queries:

            embedding = embedding_service.generate_query_embedding(
                query
            )

            entities = entities_map.get(query, [])

            candidate_roots = set()
            for entity in entities:

                roots = await engine_services.get_active_roots(
                    entity
                )

                candidate_roots.update(roots)

            fallback_used = False

            if not candidate_roots:

                roots = await engine_services.fetch_all_global_roots()

                candidate_roots.update(roots)

                fallback_used = True

            routed_payload.append({
                "query": query,
                "embedding": embedding.tolist(),
                "entities": entities,
                "candidate_roots": list(candidate_roots),
                "fallback_used": fallback_used
            })

        return routed_payload


embedding_router = EmbeddingRouter()