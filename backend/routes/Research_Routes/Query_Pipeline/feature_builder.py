class FeatureBuilder:

    def build(
        self,
        orchestration_output,
        subtrees
    ):

        features = {
            "graph_features": [],
            "risk_features": [],
            "ethics_features": [],
            "audit_features": [],
            "strategy_features": []
        }

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

        for subtree in subtrees:

            features["graph_features"].append({
                "root_node": subtree[
                    "root_node"
                ]["node_id"],
                "subtree_size": len(
                    subtree["subtree"]
                ),
                "relevance_score": subtree[
                    "relevance_score"
                ]
            })

        features[
            "strategy_features"
        ] = strategy

        features[
            "risk_features"
        ] = risk

        features[
            "ethics_features"
        ] = ethics

        features[
            "audit_features"
        ] = audit

        return features


feature_builder = FeatureBuilder()