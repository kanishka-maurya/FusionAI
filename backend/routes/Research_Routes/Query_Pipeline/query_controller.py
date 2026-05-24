from .query_expansion import query_expansion_service
from .embedding_router import embedding_router
from .parent_selector import parent_selector
from .subtree_fetcher import subtree_fetcher
from .orchestrator import mcp_orchestrator

from backend.routes.Research_Routes.Query_Pipeline.feature_builder import (
    feature_builder
)

from backend.routes.Research_Routes.Query_Pipeline.gemini_conclusion_service import (
    gemini_conclusion_service
)

from backend.routes.Research_Routes.Query_Pipeline.groq_recursive_analysis_service import (
    groq_recursive_analysis_service
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
        strategy = features[
            "strategy_features"
        ]

        selected_topic = strategy[
            "selected_topic"
        ]

        related_topics = strategy[
            "recommended_topics"
        ]

        temporal_trend = strategy[
            "temporal_trend"
        ]
        retrieved_contexts = []

        for subtree in subtrees:

            root = subtree[
                "root_node"
            ]

            retrieved_contexts.append({

                "query":
                subtree.get(
                    "query",
                    ""
                ),

                "summary":
                root.get(
                    "summary",
                    ""
                ),

                "key_points":
                root.get(
                    "key_points",
                    []
                )
            })
        gemini_output = await (
            gemini_conclusion_service
            .generate_conclusion(

                features[
                    "risk_features"
                ],

                features[
                    "ethics_features"
                ],

                features[
                    "audit_features"
                ]
            )
        )

        print(
            "\n========== GEMINI AUDIT ==========\n"
        )

        print(gemini_output)
        groq_output = await (
            groq_recursive_analysis_service
            .generate_recursive_analysis(

                selected_topic=
                selected_topic,

                temporal_trend=
                temporal_trend,

                expanded_queries=
                expanded_queries,

                retrieved_contexts=
                retrieved_contexts,

                related_topics=
                related_topics
            )
        )

        print(
            "\n========== GROQ ANALYSIS ==========\n"
        )

        print(groq_output)
        return {

            "query":
            query,

            "expanded_queries":
            expanded_queries,

            "entities":
            expanded_entities,
            "retrieved_subtrees":
            len(subtrees),

            "retrieved_contexts":
            retrieved_contexts,
            "selected_topic":
            groq_output[
                "selected_topic"
            ],

            "similar_topics":
            groq_output[
                "similar_topics"
            ],

            "topic_temporal_trend":
            groq_output[
                "topic_temporal_trend"
            ],
            "follow_up_generation":
            groq_output[
                "follow_up_generation"
            ],

            "recursive_answers":
            groq_output[
                "recursive_answers"
            ],

            "critic_mode_analysis":
            groq_output[
                "critic_mode_analysis"
            ],
            "gemini_audit":
            gemini_output,
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