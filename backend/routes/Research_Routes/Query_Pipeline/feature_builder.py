class FeatureBuilder:

    def build(
        self,
        orchestration_output,
        subtrees
    ):

        strategy = orchestration_output[
            "strategy"
        ]

        risk = orchestration_output[
            "risk"
        ] or {}

        ethics = orchestration_output[
            "ethics"
        ] or {}

        audit = orchestration_output[
            "audit"
        ] or {}

        graph_features = []

        for subtree_bundle in subtrees:

            root_node = subtree_bundle[
                "root_node"
            ]

            subtree = subtree_bundle[
                "subtree"
            ]

            graph_features.append({

                "root_node":
                    root_node["node_id"],

                "summary":
                    root_node.get(
                        "summary",
                        ""
                    ),

                "subtree_size":
                    len(subtree),

                "relevance_score":
                    subtree_bundle.get(
                        "relevance_score",
                        0
                    ),
                "score_details":
                    subtree_bundle.get(
                        "score_details",
                        {}
                    ),
                "provenance":
                    subtree_bundle.get(
                        "provenance",
                        []
                    ),

                "entities":
                    (
                        root_node.get("associated_entities")
                        or root_node.get("entities")
                        or []
                    ),

                "key_points":
                    root_node.get(
                        "key_points",
                        []
                    )
            })

        features = {

            "graph_features":
                graph_features,

            "risk_features": {

                "most_similar_pair":
                    (
                        risk.get("most_similar_pair")
                        or risk
                    )
            },

            "ethics_features": {

                "lowest_alignment_pair":
                    (
                        ethics.get("lowest_alignment_pair")
                        or ethics.get("lowest_pair")
                        or {}
                    ),
                "cluster_alignment_score":
                    ethics.get(
                        "cluster_alignment_score",
                        None
                    )
            },

            "audit_features": {

                "weakest_keypoint_alignment":
                    (
                        audit.get("weakest_keypoint_alignment")
                        or audit.get("weakest_summary_keypoint_alignment")
                        or {}
                    )
            },

            "strategy_features": {

                "selected_topic":
                    strategy.get(
                        "selected_topic",
                        {}
                    ),

                "recommended_topics":
                    strategy.get(
                        "recommended_topics",
                        []
                    ),

                "temporal_trend":
                    strategy.get(
                        "temporal_trend",
                        []
                    )
            }
        }

        return features


feature_builder = FeatureBuilder()
