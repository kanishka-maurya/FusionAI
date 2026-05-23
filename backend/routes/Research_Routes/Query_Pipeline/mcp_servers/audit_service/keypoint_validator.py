from backend.routes.Research_Routes.Query_Pipeline.mcp_servers.risk_service.risk_cross_encoder import (
    risk_cross_encoder
)


class KeypointValidator:

    def validate(self, subtrees):

        worst_pair = None
        lowest_score = float("inf")

        for subtree_bundle in subtrees:

            subtree = subtree_bundle["subtree"]

            for node in subtree:

                summary = node.get("summary", "")
                keypoints = node.get("key_points", [])

                for kp in keypoints:

                    kp_text = kp.get("text", "")

                    score = (
                        risk_cross_encoder.score(
                            summary,
                            kp_text
                        )
                    )

                    if score < lowest_score:

                        lowest_score = score

                        worst_pair = {
                            "node_id": node["node_id"],
                            "summary": summary,
                            "keypoint": kp_text,
                            "attention_score": score
                        }

        return worst_pair


keypoint_validator = KeypointValidator()