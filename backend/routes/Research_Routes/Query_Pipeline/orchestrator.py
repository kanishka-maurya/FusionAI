import asyncio
import time
print("ORCH-1 BEFORE strategy")
from backend.routes.Research_Routes.Query_Pipeline.mcp_servers.strategy_service.strategy_server import (
    strategy_service
)
print("ORCH-1 DONE")

print("ORCH-2 BEFORE risk")
from backend.routes.Research_Routes.Query_Pipeline.mcp_servers.risk_service.risk_scoring import (
    risk_service
)
print("ORCH-2 DONE")

print("ORCH-3 BEFORE ethics")
from backend.routes.Research_Routes.Query_Pipeline.mcp_servers.ethics_service.ethics_validator import (
    ethics_service
)
print("ORCH-3 DONE")

print("ORCH-4 BEFORE audit")
from backend.routes.Research_Routes.Query_Pipeline.mcp_servers.audit_service.audit_server import (
    audit_service
)
print("ORCH-4 DONE")

class MCPOrchestrator:

    async def execute(self, subtrees):

        print(
            "\n[ORCHESTRATOR] Starting execute: "
            f"{len(subtrees)} subtrees"
        )

        parent_nodes = [
            subtree["root_node"]
            for subtree in subtrees
        ]

        async def run_service(name, coroutine, fallback, timeout=75):

            started_at = time.perf_counter()

            print(
                f"\n[ORCHESTRATOR] {name} started"
            )

            try:

                result = await asyncio.wait_for(
                    coroutine,
                    timeout=timeout
                )

                elapsed = time.perf_counter() - started_at

                print(
                    f"\n[ORCHESTRATOR] {name} completed "
                    f"in {elapsed:.2f}s"
                )

                return result

            except asyncio.TimeoutError:

                elapsed = time.perf_counter() - started_at

                print(
                    f"\n[ORCHESTRATOR TIMEOUT] {name} exceeded "
                    f"{timeout}s after {elapsed:.2f}s; "
                    "continuing with fallback"
                )

                return fallback

            except Exception as e:

                elapsed = time.perf_counter() - started_at

                print(
                    f"\n[ORCHESTRATOR ERROR] {name} failed "
                    f"after {elapsed:.2f}s:",
                    repr(e)
                )

                return fallback

        strategy_task = run_service(
            "strategy",
            strategy_service.process(subtrees),
            {
                "selected_topic": {
                    "topic": None,
                    "frequency": 0,
                    "depth": 0,
                    "score": 0
                },
                "recommended_topics": [],
                "temporal_trend": []
            },
            timeout=30
        )

        risk_task = run_service(
            "risk",
            risk_service.process(parent_nodes),
            None,
            timeout=75
        )

        ethics_task = run_service(
            "ethics",
            ethics_service.process(subtrees),
            None,
            timeout=75
        )

        audit_task = run_service(
            "audit",
            audit_service.process(subtrees),
            {
                "weakest_summary_keypoint_alignment": None,
                "traceability": None
            },
            timeout=75
        )

        strategy_result, risk_result, ethics_result, audit_result = (
            await asyncio.gather(
                strategy_task,
                risk_task,
                ethics_task,
                audit_task
            )
        )

        print(
            "\n[ORCHESTRATOR] All services completed"
        )

        return {
            "strategy": strategy_result,
            "risk": risk_result,
            "ethics": ethics_result,
            "audit": audit_result
        }


mcp_orchestrator = MCPOrchestrator()
