import asyncio
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

        parent_nodes = [
            subtree["root_node"]
            for subtree in subtrees
        ]

        strategy_task = strategy_service.process(
            subtrees
        )

        risk_task = risk_service.process(
            parent_nodes
        )

        ethics_task = ethics_service.process(
            subtrees
        )

        audit_task = audit_service.process(
            subtrees
        )

        strategy_result, risk_result, ethics_result, audit_result = (
            await asyncio.gather(
                strategy_task,
                risk_task,
                ethics_task,
                audit_task
            )
        )

        return {
            "strategy": strategy_result,
            "risk": risk_result,
            "ethics": ethics_result,
            "audit": audit_result
        }


mcp_orchestrator = MCPOrchestrator()