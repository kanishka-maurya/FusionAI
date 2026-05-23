from .keypoint_validator import keypoint_validator
from .traceability import traceability


class AuditService:

    async def process(self, subtrees):

        weakest_alignment = (
            keypoint_validator.validate(subtrees)
        )

        trace = None

        if weakest_alignment:

            node_id = weakest_alignment["node_id"]

            for subtree_bundle in subtrees:

                subtree = subtree_bundle["subtree"]

                for node in subtree:

                    if node["node_id"] == node_id:

                        trace = traceability.trace(subtree)
                        break

        return {
            "weakest_summary_keypoint_alignment":
                weakest_alignment,

            "traceability": trace
        }


audit_service = AuditService()