from .semantic_alignment import semantic_alignment


class EthicsService:

    async def process(self, subtrees):

        violations = []

        for subtree in subtrees:

            nodes = subtree["subtree"]

            for i in range(len(nodes)):

                for j in range(i + 1, len(nodes)):

                    n1 = nodes[i]
                    n2 = nodes[j]

                    score = semantic_alignment.compare(
                        n1["summary"],
                        n2["summary"]
                    )

                    if score < 0.3:

                        violations.append({
                            "node_1": n1["node_id"],
                            "node_2": n2["node_id"],
                            "attention_score": score,
                            "issue": "semantic misalignment"
                        })

        return violations

ethics_service = EthicsService()