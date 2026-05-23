from .query_expansion import query_expansion_service
from .embedding_router import embedding_router
from .parent_selector import parent_selector
from .subtree_fetcher import subtree_fetcher
from .orchestrator import mcp_orchestrator

from backend.routes.Research_Routes.Query_Pipeline.feature_builder import (
    feature_builder
)


class QueryController:

    async def process(self, query):
        expanded = await (
            query_expansion_service.expand_query(
                query
            )
        )

        expanded_queries = expanded[
            "queries"
        ]

        expanded_entities = expanded[
            "entities"
        ]
        routed_queries = await (
            embedding_router.route_queries(
                expanded_queries,
                expanded_entities
            )
        )
        selected_parents = []

        for routed in routed_queries:

            selected = await (
                parent_selector.select_best_parent(
                    routed
                )
            )

            if selected:
                selected_parents.append(
                    selected
                )

        subtrees = []

        for parent in selected_parents:

            subtree = await (
                subtree_fetcher.fetch(
                    parent
                )
            )

            if subtree:

                subtrees.append(
                    subtree
                )

        orchestration_output = await (
            mcp_orchestrator.execute(
                subtrees
            )
        )
        features = feature_builder.build(
            orchestration_output,
            subtrees
        )
        return {

            "query": query,

            "expanded_queries":
                expanded_queries,

            "entities":
                expanded_entities,

            "retrieved_subtrees":
                len(subtrees),

            "subtrees":
                subtrees,
            "features": {
                "graph_features":
                    features[
                        "graph_features"
                    ],
                "risk_analysis":
                    features[
                        "risk_features"
                    ],
                "ethics_analysis":
                    features[
                        "ethics_features"
                    ],
                "audit_analysis":
                    features[
                        "audit_features"
                    ],
                "strategy_analysis":
                    features[
                        "strategy_features"
                    ]
            }
        }


query_controller = QueryController()