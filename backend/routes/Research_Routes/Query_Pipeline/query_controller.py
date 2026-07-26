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

try:
    print("QC-8B BEFORE groq_final_answer_service")
    from backend.routes.Research_Routes.Query_Pipeline.groq_final_answer_service import (
        groq_final_answer_service
    )
    print("QC-8B DONE")
except Exception as e:
    print("QC-8B ERROR:", repr(e))
    raise

try:
    print("QC-8C BEFORE web_evidence_service")
    from backend.routes.Research_Routes.Query_Pipeline.web_evidence_service import (
        web_evidence_service
    )
    print("QC-8C DONE")
except Exception as e:
    print("QC-8C ERROR:", repr(e))
    raise

from backend.routes.Research_Routes.utils import (
    detect_contradictions,
    lexical_overlap,
    node_text,
    retrieval_quality_report,
)
from backend.routes.Research_Routes.graph_maintenance import (
    graph_health_report,
)

print("QC-9 BEFORE CLASS")

MIN_RELEVANCE_SCORE = 0.28
MIN_LEXICAL_OVERLAP = 0.20
MIN_KEYWORD_SCORE = 0.20
MIN_STRONG_QUERY_COVERAGE = 0.45
MIN_STRONG_AVERAGE_RELEVANCE = 0.40
PIPELINE_DIAGNOSTIC_TERMS = [
    "pipeline",
    "audit",
    "diagnostic",
    "retrieval quality",
    "risk features",
    "ethics features",
    "summary-keypoint",
    "system developers",
    "feature builder",
]


def _topic_label(topic):

    if isinstance(topic, dict):
        return topic.get("topic") or "No dominant topic found"

    if topic:
        return str(topic)

    return "No dominant topic found"


def _looks_like_pipeline_diagnostic(text):

    lowered = str(text or "").lower()

    return any(
        term in lowered
        for term in PIPELINE_DIAGNOSTIC_TERMS
    )


def _evidence_based_answer(query, retrieved_contexts):

    evidence_lines = []

    for index, ctx in enumerate(retrieved_contexts[:4], start=1):

        summary = str(ctx.get("summary") or "").strip()

        if not summary:
            continue

        evidence_lines.append(
            f"[Evidence {index}] {summary}"
        )

    if not evidence_lines:

        return (
            "FusionAI could not retrieve enough relevant evidence to answer "
            f"'{query}' confidently. Try a more specific AI-news query or "
            "refresh the intelligence feed before asking again."
        )

    return (
        f"For '{query}', FusionAI retrieved the following relevant evidence: "
        + " ".join(evidence_lines)
    )


def _needs_web_reinforcement(query, retrieval_quality, retrieved_contexts):

    normalized_query = str(query or "").strip().lower()

    broad_intent_markers = [
        "what is",
        "tell me about",
        "explain",
        "difference between",
        "compare",
        "vs",
        "versus",
        "latest",
        "updates",
        "recent",
    ]

    has_broad_intent = any(
        marker in normalized_query
        for marker in broad_intent_markers
    )

    weak_coverage = (
        float(retrieval_quality.get("query_coverage") or 0.0)
        < MIN_STRONG_QUERY_COVERAGE
    )

    weak_relevance = (
        float(retrieval_quality.get("average_relevance_score") or 0.0)
        < MIN_STRONG_AVERAGE_RELEVANCE
    )

    sparse_context = len(retrieved_contexts) < 3

    direct_miss = any(
        "not directly" in str(ctx.get("summary") or "").lower()
        or "not clearly" in str(ctx.get("summary") or "").lower()
        for ctx in retrieved_contexts
    )

    return (
        has_broad_intent
        or weak_coverage
        or weak_relevance
        or sparse_context
        or direct_miss
    )


def _passes_relevance_gate(subtree):

    score_details = subtree.get("score_details", {})
    relevance_score = float(
        subtree.get("relevance_score")
        or score_details.get("hybrid_score")
        or 0.0
    )
    lexical_score = float(
        score_details.get("lexical_score")
        or 0.0
    )
    keyword_score = float(
        score_details.get("keyword_score")
        or 0.0
    )
    entity_score = float(
        score_details.get("entity_score")
        or 0.0
    )

    text_overlap = lexical_overlap(
        subtree.get("query", ""),
        node_text(subtree.get("root_node", {}))
    )

    return (
        relevance_score >= MIN_RELEVANCE_SCORE
        and (
            lexical_score >= MIN_LEXICAL_OVERLAP
            or keyword_score >= MIN_KEYWORD_SCORE
            or entity_score > 0
            or text_overlap >= MIN_LEXICAL_OVERLAP
        )
    )


def _filter_relevant_subtrees(subtrees):

    if not subtrees:
        return [], []

    relevant = []
    rejected = []

    for subtree in subtrees:

        if _passes_relevance_gate(subtree):
            relevant.append(subtree)
        else:
            rejected.append({
                "query": subtree.get("query", ""),
                "root_id": (
                    subtree
                    .get("root_node", {})
                    .get("node_id")
                ),
                "summary": (
                    subtree
                    .get("root_node", {})
                    .get("summary", "")
                )[:220],
                "score_details": subtree.get(
                    "score_details",
                    {}
                )
            })

    if relevant:
        return relevant, rejected

    best_subtree = max(
        subtrees,
        key=lambda item: float(
            item.get("relevance_score")
            or item.get("score_details", {}).get("hybrid_score")
            or 0.0
        )
    )

    return [best_subtree], rejected


def _build_user_view(
    *,
    query,
    expanded_queries,
    retrieved_contexts,
    retrieval_quality,
    graph_health,
    contradictions,
    gemini_output,
    groq_output,
    final_answer_output=None
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

    final_answer_output = final_answer_output or {}

    final_answer = final_answer_output.get("answer")

    if (
        final_answer
        and not _looks_like_pipeline_diagnostic(final_answer)
    ):
        main_summary = final_answer_output.get(
            "answer",
            ""
        )
    elif (
        answers
        and answers[0].get("answer")
        and not _looks_like_pipeline_diagnostic(
            answers[0].get("answer", "")
        )
    ):
        main_summary = answers[0].get(
            "answer",
            ""
        )
    else:
        main_summary = _evidence_based_answer(
            query,
            retrieved_contexts
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
        "recommended_searches": (
            final_answer_output.get("recommended_searches")
            or [
                item["search_query"]
                for item in similar_topics
            ]
        ),
        "key_findings": final_answer_output.get(
            "key_findings",
            []
        ),
        "evidence_used": final_answer_output.get(
            "evidence_used",
            []
        ),
        "limitations": final_answer_output.get(
            "limitations",
            []
        ),
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

        subtrees, rejected_subtrees = _filter_relevant_subtrees(
            subtrees
        )

        print(
            "\n[PIPELINE] Relevance gate kept "
            f"{len(subtrees)} subtrees and rejected "
            f"{len(rejected_subtrees)} low-signal subtrees"
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

        web_evidence = []

        if _needs_web_reinforcement(
            query,
            retrieval_quality,
            retrieved_contexts
        ):

            print(
                "\n[PIPELINE] Retrieval needs live web reinforcement"
            )

            web_evidence = await web_evidence_service.search(
                query
            )

            if web_evidence:

                retrieved_contexts = (
                    web_evidence
                    + retrieved_contexts
                )

                retrieval_quality = retrieval_quality_report(
                    query,
                    expanded_queries,
                    [
                        *[
                            {
                                "root_node": {
                                    "summary": ctx.get("summary", ""),
                                    "key_points": ctx.get("key_points", [])
                                },
                                "relevance_score": (
                                    ctx
                                    .get("score_details", {})
                                    .get("hybrid_score", 0.0)
                                ),
                                "fallback_used": False
                            }
                            for ctx in web_evidence
                        ],
                        *subtrees
                    ]
                )

                contradictions = detect_contradictions(
                    retrieved_contexts
                )

                print(
                    "\n[PIPELINE] Live web evidence injected: "
                    f"{len(web_evidence)} items"
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
                "web_evidence_count": len(web_evidence),
                "rejected_subtrees": rejected_subtrees,
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
            "\n[PIPELINE] Starting final answer synthesis"
        )

        final_answer_output = await (
            groq_final_answer_service
            .generate_answer(
                query=query,
                selected_topic=selected_topic,
                related_topics=groq_output.get(
                    "similar_topics",
                    related_topics
                ),
                expanded_queries=expanded_queries,
                retrieved_contexts=retrieved_contexts,
                audit_summary={
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
                    )[:3],
                    "retrieval_quality": retrieval_quality,
                    "graph_health": graph_health,
                    "contradictions": contradictions
                }
            )
        )

        print(
            "\n[PIPELINE] Final answer synthesis completed"
        )

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
            groq_output=groq_output,
            final_answer_output=final_answer_output
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
            "web_evidence_count":
            len(web_evidence),
            "rejected_subtrees":
            rejected_subtrees,
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
            "final_answer":
            final_answer_output,
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
