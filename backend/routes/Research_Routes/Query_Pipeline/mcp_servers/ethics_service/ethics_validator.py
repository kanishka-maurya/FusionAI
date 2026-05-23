from .semantic_alignment import semantic_alignment


class EthicsService:

    async def process(self, subtrees):

        lowest_alignment = None

        for subtree_bundle in subtrees:

            nodes = subtree_bundle["subtree"]

            if len(nodes) < 2:
                continue

            cluster_score = 0.0
            pair_count = 0

            local_lowest = None

            for i in range(len(nodes)):

                for j in range(i + 1, len(nodes)):

                    n1 = nodes[i]
                    n2 = nodes[j]

                    attention_score = (
                        semantic_alignment.compare(
                            n1["summary"],
                            n2["summary"]
                        )
                    )

                    cluster_score += attention_score
                    pair_count += 1

                    if (
                        local_lowest is None or
                        attention_score < local_lowest["attention_score"]
                    ):

                        local_lowest = {
                            "node_1": n1["node_id"],
                            "node_2": n2["node_id"],
                            "summary_1": n1["summary"],
                            "summary_2": n2["summary"],
                            "attention_score": attention_score
                        }

            if pair_count == 0:
                continue

            avg_cluster_alignment = (
                cluster_score / pair_count
            )

            if (
                lowest_alignment is None or
                avg_cluster_alignment <
                lowest_alignment["cluster_alignment_score"]
            ):

                lowest_alignment = {
                    "cluster_alignment_score":
                        avg_cluster_alignment,

                    "lowest_pair":
                        local_lowest
                }

        return lowest_alignment


ethics_service = EthicsService()