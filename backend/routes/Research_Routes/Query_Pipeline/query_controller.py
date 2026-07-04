print("QC-1 BEFORE query_expansion")
from .query_expansion import query_expansion_service
print("QC-1 DONE")

print("QC-2 BEFORE embedding_router")
from .embedding_router import embedding_router
print("QC-2 DONE")

print("QC-3 BEFORE parent_selector")
from .parent_selector import parent_selector
print("QC-3 DONE")

print("QC-4 BEFORE subtree_fetcher")
from .subtree_fetcher import subtree_fetcher
print("QC-4 DONE")

try:
    print("QC-5 BEFORE orchestrator")
    from .orchestrator import mcp_orchestrator
    print("QC-5 DONE")
except Exception as e:
    print("QC-5 ERROR:", repr(e))
    raise

try:
    print("QC-6 BEFORE feature_builder")
    from backend.routes.Research_Routes.Query_Pipeline.feature_builder import (
        feature_builder
    )
    print("QC-6 DONE")
except Exception as e:
    print("QC-6 ERROR:", repr(e))
    raise

try:
    print("QC-7 BEFORE gemini_conclusion_service")
    from backend.routes.Research_Routes.Query_Pipeline.gemini_conclusion_service import (
        gemini_conclusion_service
    )
    print("QC-7 DONE")
except Exception as e:
    print("QC-7 ERROR:", repr(e))
    raise

try:
    print("QC-8 BEFORE groq_recursive_analysis_service")
    from backend.routes.Research_Routes.Query_Pipeline.groq_recursive_analysis_service import (
        groq_recursive_analysis_service
    )
    print("QC-8 DONE")
except Exception as e:
    print("QC-8 ERROR:", repr(e))
    raise

from backend.routes.Research_Routes.utils import (
    detect_contradictions,
    retrieval_quality_report,
)
from backend.routes.Research_Routes.graph_maintenance import (
    graph_health_report,
)

print("QC-9 BEFORE CLASS")

def _topic_label(topic):

    if isinstance(topic, dict):
        return topic.get("topic") or "No dominant topic found"

    if topic:
        return str(topic)

    return "No dominant topic found"


def _build_user_view(
    *,
    query,
    expanded_queries,
    retrieved_contexts,
    retrieval_quality,
    graph_health,
    contradictions,
    gemini_output,
    groq_output
):

    selected_topic = groq_output.get(
        "selected_topic",
        {}
    )

    topic_name = _topic_label(
        selected_topic
    )

    similar_topics = []

    for item in groq_output.get("similar_topics", [])[:8]:

        if isinstance(item, dict):

            topic = item.get("topic")

            if not topic:
                continue

            similar_topics.append({
                "topic": topic,
                "search_query": topic,
                "reason": (
                    "Frequently appears with "
                    f"{topic_name} in the retrieved graph."
                ),
                "co_occurrence_count": item.get(
                    "co_occurrence_count",
                    0
                )
            })

        elif item:

            similar_topics.append({
                "topic": str(item),
                "search_query": str(item),
                "reason": (
                    "Related to the selected research topic."
                ),
                "co_occurrence_count": 0
            })

    answers = (
        groq_output
        .get("recursive_answers", {})
        .get("generated_answers", [])
    )

    followups = (
        groq_output
        .get("follow_up_generation", {})
        .get("follow_up_questions", [])
    )

    if answers:
        main_summary = answers[0].get(
            "answer",
            ""
        )
    else:
        main_summary = (
            gemini_output.get("final_conclusion")
            or groq_output
            .get("critic_mode_analysis", {})
            .get("final_intelligence_summary")
            or (
                "FusionAI retrieved relevant graph context and related "
                "topics, but recursive answer generation was unavailable "
                "for this query."
            )
        )

    evidence = []

    for ctx in retrieved_contexts[:5]:

        evidence.append({
            "query": ctx.get("query", ""),
            "summary": ctx.get("summary", ""),
            "key_points": ctx.get("key_points", [])[:4],
            "provenance": ctx.get("provenance", [])
        })

    return {
        "query": query,
        "topic": selected_topic,
        "topic_name": topic_name,
        "summary": main_summary,
        "similar_topics": similar_topics,
        "suggested_searches": [
            item["search_query"]
            for item in similar_topics
        ],
        "follow_up_questions": followups,
        "answers": answers,
        "audit_summary": {
            "overall_assessment": gemini_output.get(
                "overall_assessment",
                ""
            ),
            "final_conclusion": gemini_output.get(
                "final_conclusion",
                ""
            ),
            "detected_issues": gemini_output.get(
                "detected_issues",
                []
            )[:5],
            "retrieval_quality": retrieval_quality,
            "graph_health": graph_health,
            "contradictions": contradictions
        },
        "retrieved_evidence": evidence,
        "expanded_queries": expanded_queries[:5]
    }


class QueryController:

    async def process(self, query, mode="deep_research"):

        print(
            "\n[PIPELINE] Starting query expansion"
        )

        expanded = await (
            query_expansion_service.expand_query(
                query
            )
        )

        expanded_queries = (
            expanded.get("queries")
            if isinstance(expanded, dict)
            else None
        )

        if not expanded_queries:

            expanded_queries = [query]

        expanded_entities = (
            expanded.get("entities")
            if isinstance(expanded, dict)
            else None
        )

        if not isinstance(expanded_entities, dict):

            expanded_entities = {
                query: []
            }

        print(
            "\n[PIPELINE] Query expansion completed: "
            f"{len(expanded_queries)} queries"
        )

        print(
            "\n[PIPELINE] Starting embedding routing"
        )
        routed_queries = await (
            embedding_router.route_queries(
                expanded_queries,
                expanded_entities
            )
        )
        print(
            "\n[PIPELINE] Embedding routing completed: "
            f"{len(routed_queries)} routed queries"
        )

        selected_parents = []

        print(
            "\n[PIPELINE] Starting parent selection"
        )

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

        print(
            "\n[PIPELINE] Parent selection completed: "
            f"{len(selected_parents)} selected parents"
        )

        subtrees = []

        print(
            "\n[PIPELINE] Starting subtree fetch"
        )

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

        print(
            "\n[PIPELINE] Subtree fetch completed: "
            f"{len(subtrees)} subtrees"
        )

        print(
            "\n[PIPELINE] Starting orchestration"
        )

        orchestration_output = await (
            mcp_orchestrator.execute(
                subtrees
            )
        )

        print(
            "\n[PIPELINE] Orchestration completed"
        )

        print(
            "\n[PIPELINE] Building features"
        )

        features = feature_builder.build(
            orchestration_output,
            subtrees
        )

        print(
            "\n[PIPELINE] Feature building completed"
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
                ),

                "provenance":
                subtree.get(
                    "provenance",
                    []
                ),

                "score_details":
                subtree.get(
                    "score_details",
                    {}
                )
            })
        retrieval_quality = retrieval_quality_report(
            query,
            expanded_queries,
            subtrees
        )
        graph_health = graph_health_report(
            subtrees
        )
        contradictions = detect_contradictions(
            retrieved_contexts
        )
        if mode == "retrieval_only":
            print(
                "\n[PIPELINE] Returning retrieval-only response"
            )

            return {
                "query": query,
                "research_mode": mode,
                "expanded_queries": expanded_queries,
                "entities": expanded_entities,
                "retrieved_subtrees": len(subtrees),
                "retrieved_contexts": retrieved_contexts,
                "retrieval_quality": retrieval_quality,
                "graph_health": graph_health,
                "contradiction_scan": contradictions,
                "provenance": [
                    subtree.get("provenance", [])
                    for subtree in subtrees
                ],
                "features": features
            }

        print(
            "\n[PIPELINE] Starting Gemini conclusion service"
        )

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

        print(
            "\n[PIPELINE] Starting Groq recursive analysis service"
        )

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

        print(
            "\n[PIPELINE] Returning deep research response"
        )

        user_view = _build_user_view(
            query=query,
            expanded_queries=expanded_queries,
            retrieved_contexts=retrieved_contexts,
            retrieval_quality=retrieval_quality,
            graph_health=graph_health,
            contradictions=contradictions,
            gemini_output=gemini_output,
            groq_output=groq_output
        )

        return {

            "query":
            query,
            "research_mode":
            mode,

            "expanded_queries":
            expanded_queries,

            "entities":
            expanded_entities,
            "retrieved_subtrees":
            len(subtrees),

            "retrieved_contexts":
            retrieved_contexts,
            "retrieval_quality":
            retrieval_quality,
            "graph_health":
            graph_health,
            "contradiction_scan":
            contradictions,
            "provenance":
            [
                subtree.get("provenance", [])
                for subtree in subtrees
            ],
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
            "user_view":
            user_view,
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


print("QC-10 BEFORE INSTANCE")
query_controller = QueryController()
print("QC-11 INSTANCE CREATED")
