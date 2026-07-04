from collections import defaultdict
import ast
import time

from .topic_cube import topic_cube
from .temporal_trends import temporal_trend_analyzer
from .topic_depth import topic_depth_analyzer
from .cooccurrence import cooccurrence_analyzer


class StrategyService:

    async def process(self, subtrees):

        started_at = time.perf_counter()

        print(
            "\n[STRATEGY SERVICE] Starting: "
            f"{len(subtrees)} subtrees"
        )

        cube = topic_cube.build_topic_cube(
            subtrees
        )

        print(
            "\n[STRATEGY SERVICE] Topic cube built: "
            f"{len(cube)} entities"
        )

        temporal_trends = (
            temporal_trend_analyzer.analyze(
                cube
            )
        )

        print(
            "\n[STRATEGY SERVICE] Temporal trends analyzed"
        )

        cooccurrence = (
            cooccurrence_analyzer.analyze(
                subtrees
            )
        )

        print(
            "\n[STRATEGY SERVICE] Co-occurrence analyzed: "
            f"{len(cooccurrence)} pairs"
        )

        entity_frequency = defaultdict(int)

        entity_depth = defaultdict(int)

        for subtree_bundle in subtrees:

            subtree = subtree_bundle["subtree"]

            depth_map = (
                topic_depth_analyzer.calculate_depth(
                    subtree_bundle
                )
            )

            entities_seen = set()

            for node in subtree:

                entities = (
                    node.get("associated_entities")
                    or node.get("entities")
                    or []
                )

                for entity in entities:

                    if entity not in entities_seen:

                        entity_frequency[
                            entity
                        ] += 1

                        entities_seen.add(
                            entity
                        )

                    entity_depth[
                        entity
                    ] += depth_map.get(
                        entity,
                        1
                    )

        best_topic = None

        best_score = -1

        for entity in entity_frequency:

            freq_score = (
                entity_frequency[entity]
            )

            depth_score = (
                entity_depth[entity]
            )

            final_score = (
                (0.7 * freq_score) +
                (0.3 * depth_score)
            )

            if final_score > best_score:

                best_score = final_score

                best_topic = entity

        related_topics = []

        for pair, stats in cooccurrence.items():

            try:
                e1, e2 = ast.literal_eval(pair)
            except (SyntaxError, ValueError, TypeError):
                continue

            if best_topic not in [e1, e2]:
                continue

            related_topic = (
                e2 if e1 == best_topic
                else e1
            )

            related_topics.append({

                "topic": related_topic,

                "co_occurrence_count":
                    stats[
                        "co_occurrence_count"
                    ],

                "total_occurrences":
                    stats[
                        "total"
                    ]
            })

        related_topics = sorted(
            related_topics,
            key=lambda x: (
                x["co_occurrence_count"],
                x["total_occurrences"]
            ),
            reverse=True
        )

        selected_temporal_trend = (
            temporal_trends.get(
                best_topic,
                []
            )
        )

        elapsed = time.perf_counter() - started_at

        print(
            "\n[STRATEGY SERVICE] Completed in "
            f"{elapsed:.2f}s with selected topic: "
            f"{best_topic}"
        )

        return {

            "selected_topic": {

                "topic": best_topic,

                "frequency":
                    entity_frequency.get(
                        best_topic,
                        0
                    ),

                "depth":
                    entity_depth.get(
                        best_topic,
                        0
                    ),

                "score":
                    best_score
            },

            "recommended_topics":
                related_topics[:10],

            "temporal_trend":
                selected_temporal_trend
        }


strategy_service = StrategyService()
