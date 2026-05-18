from backend.routes.Research_Routes.Nexus_Graph_DB.services import (
    engine_services
)


class KeypointValidator:

    def validate(self, node):

        kps = node.get("key_points", [])

        consistency_scores = []

        for i in range(len(kps)):

            for j in range(i + 1, len(kps)):

                score = engine_services.cosine_similarity(
                    kps[i]["kp_embedding"],
                    kps[j]["kp_embedding"]
                )

                consistency_scores.append(score)

        return consistency_scores


keypoint_validator = KeypointValidator()