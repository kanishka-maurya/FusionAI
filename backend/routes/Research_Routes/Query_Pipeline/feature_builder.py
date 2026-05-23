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
        ]

        ethics = orchestration_output[
            "ethics"
        ]

        audit = orchestration_output[
            "audit"
        ]

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

                "entities":
                    root_node.get(
                        "entities",
                        []
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
                    risk.get(
                        "most_similar_pair",
                        {}
                    )
            },

            "ethics_features": {

                "lowest_alignment_pair":
                    ethics.get(
                        "lowest_alignment_pair",
                        {}
                    )
            },

            "audit_features": {

                "weakest_keypoint_alignment":
                    audit.get(
                        "weakest_keypoint_alignment",
                        {}
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