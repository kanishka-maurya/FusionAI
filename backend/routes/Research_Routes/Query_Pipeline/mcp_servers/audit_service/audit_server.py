from .keypoint_validator import keypoint_validator
from .traceability import traceability


class AuditService:

    async def process(self, subtrees):

        payload = []

        for subtree_bundle in subtrees:

            subtree = subtree_bundle["subtree"]

            for node in subtree:

                payload.append({
                    "node_id": node["node_id"],
                    "keypoint_consistency": (
                        keypoint_validator.validate(node)
                    ),
                    "traceability": (
                        traceability.trace(subtree)
                    )
                })

        return payload


audit_service = AuditService()