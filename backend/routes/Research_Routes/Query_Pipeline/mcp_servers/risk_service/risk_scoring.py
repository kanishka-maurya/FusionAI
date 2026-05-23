from .pair_builder import pair_builder
from .risk_cross_encoder import risk_cross_encoder


class RiskService:

    async def process(self, parent_nodes):

        pairs = pair_builder.build_pairs(
            parent_nodes
        )

        highest_pair = None
        highest_score = float("-inf")

        for node1, node2 in pairs:

            attention = risk_cross_encoder.score(
                node1["summary"],
                node2["summary"]
            )

            # keep only highest positive attention pair
            if attention > highest_score:

                highest_score = attention

                highest_pair = {
                    "node_1": node1["node_id"],
                    "node_2": node2["node_id"],
                    "summary_1": node1["summary"],
                    "summary_2": node2["summary"],
                    "attention_score": attention
                }

        return highest_pair


risk_service = RiskService()