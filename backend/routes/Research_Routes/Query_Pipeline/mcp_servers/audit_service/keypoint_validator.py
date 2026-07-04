from backend.routes.Research_Routes.Query_Pipeline.mcp_servers.risk_service.risk_cross_encoder import (
    risk_cross_encoder
)
import time


class KeypointValidator:

    def validate(self, subtrees):

        started_at = time.perf_counter()

        print(
            "\n[KEYPOINT VALIDATOR] Starting"
        )

        worst_pair = None
        lowest_score = float("inf")
        scored_count = 0

        for subtree_index, subtree_bundle in enumerate(subtrees, start=1):

            subtree = subtree_bundle["subtree"]

            print(
                "\n[KEYPOINT VALIDATOR] Subtree "
                f"{subtree_index}/{len(subtrees)} has "
                f"{len(subtree)} nodes"
            )

            for node_index, node in enumerate(subtree, start=1):

                summary = node.get("summary", "")
                keypoints = node.get("key_points", [])

                for keypoint_index, kp in enumerate(keypoints, start=1):

                    kp_text = kp.get("text", "")

                    scored_count += 1

                    print(
                        "\n[KEYPOINT VALIDATOR] Scoring subtree "
                        f"{subtree_index}, node {node_index}, "
                        f"keypoint {keypoint_index}/{len(keypoints)}"
                    )

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

        elapsed = time.perf_counter() - started_at

        print(
            "\n[KEYPOINT VALIDATOR] Completed "
            f"{scored_count} scores in {elapsed:.2f}s"
        )

        return worst_pair


keypoint_validator = KeypointValidator()
