print("RISK-1 BEFORE pair_builder")
from .pair_builder import pair_builder
print("RISK-1 DONE")

print("RISK-2 BEFORE risk_cross_encoder")
from .risk_cross_encoder import risk_cross_encoder
print("RISK-2 DONE")
import asyncio
import time

class RiskService:

    async def process(self, parent_nodes):

        return await asyncio.to_thread(
            self._process_sync,
            parent_nodes
        )

    def _process_sync(self, parent_nodes):

        started_at = time.perf_counter()

        print(
            "\n[RISK SERVICE] Starting: "
            f"{len(parent_nodes)} parent nodes"
        )

        pairs = pair_builder.build_pairs(
            parent_nodes
        )

        print(
            "\n[RISK SERVICE] Built pairs: "
            f"{len(pairs)}"
        )

        highest_pair = None
        highest_score = float("-inf")

        for index, (node1, node2) in enumerate(pairs, start=1):

            print(
                "\n[RISK SERVICE] Scoring pair "
                f"{index}/{len(pairs)}"
            )

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

        elapsed = time.perf_counter() - started_at

        print(
            "\n[RISK SERVICE] Completed in "
            f"{elapsed:.2f}s"
        )

        return highest_pair


risk_service = RiskService()
