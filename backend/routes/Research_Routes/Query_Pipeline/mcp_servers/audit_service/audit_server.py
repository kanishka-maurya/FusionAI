from .keypoint_validator import keypoint_validator
from .traceability import traceability
import asyncio
import time


class AuditService:

    async def process(self, subtrees):

        return await asyncio.to_thread(
            self._process_sync,
            subtrees
        )

    def _process_sync(self, subtrees):

        started_at = time.perf_counter()

        print(
            "\n[AUDIT SERVICE] Starting: "
            f"{len(subtrees)} subtrees"
        )

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

        elapsed = time.perf_counter() - started_at

        print(
            "\n[AUDIT SERVICE] Completed in "
            f"{elapsed:.2f}s"
        )

        return {
            "weakest_summary_keypoint_alignment":
                weakest_alignment,

            "traceability": trace
        }


audit_service = AuditService()
