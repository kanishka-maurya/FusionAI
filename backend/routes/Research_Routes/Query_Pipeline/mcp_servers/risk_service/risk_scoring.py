from .pair_builder import pair_builder
from .risk_cross_encoder import risk_cross_encoder


class RiskService:

    async def process(self, parent_nodes):

        pairs = pair_builder.build_pairs(
            parent_nodes
        )

        payload = []

        for node1, node2 in pairs:

            attention = risk_cross_encoder.score(
                node1["summary"],
                node2["summary"]
            )

            payload.append({
                "node_1": node1["node_id"],
                "node_2": node2["node_id"],
                "attention_score": attention
            })

        return payload


risk_service = RiskService()