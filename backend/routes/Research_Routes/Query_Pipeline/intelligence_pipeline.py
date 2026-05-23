from backend.routes.Research_Routes.Query_Pipeline.groq_recursive_analysis_service import (
    groq_recursive_analysis_service
)

from backend.routes.Research_Routes.Query_Pipeline.gemini_conclusion_service import (
    gemini_conclusion_service
)


class IntelligencePipeline:

    async def generate(
        self,
        features,
        expanded_queries,
        subtrees
    ):

        strategy = (
            features["strategy_features"]
        )

        selected_topic = (
            strategy["selected_topic"]
        )

        related_topics = (
            strategy["recommended_topics"]
        )

        temporal_trend = (
            strategy["temporal_trend"]
        )

        retrieved_contexts = []

        for subtree in subtrees:

            root = subtree["root_node"]

            retrieved_contexts.append({

                "query":
                subtree.get("query", ""),

                "summary":
                root.get("summary", ""),

                "key_points":
                root.get("key_points", [])
            })

        gemini_output = await (
            gemini_conclusion_service
            .generate_conclusion(
                features["risk_features"],
                features["ethics_features"],
                features["audit_features"]
            )
        )

        print("\n========== GEMINI AUDIT ==========\n")
        print(gemini_output)

        groq_output = await (
            groq_recursive_analysis_service
            .generate_recursive_analysis(
                selected_topic=selected_topic,

                temporal_trend=temporal_trend,

                expanded_queries=(
                    expanded_queries
                ),

                retrieved_contexts=(
                    retrieved_contexts
                ),

                related_topics=(
                    related_topics
                )
            )
        )

        print("\n========== GROQ ANALYSIS ==========\n")
        print(groq_output)

        return {

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
            ]
        }


intelligence_pipeline = (
    IntelligencePipeline()
)