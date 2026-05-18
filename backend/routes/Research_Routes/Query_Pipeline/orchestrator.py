import asyncio

from backend.routes.Research_Routes.Query_Pipeline.mcp_servers.strategy_service.strategy_server import (
    strategy_service
)

from backend.routes.Research_Routes.Query_Pipeline.mcp_servers.risk_service.risk_scoring import (
    risk_service
)

from backend.routes.Research_Routes.Query_Pipeline.mcp_servers.ethics_service.ethics_validator import (
    ethics_service
)

from backend.routes.Research_Routes.Query_Pipeline.mcp_servers.audit_service.audit_server import (
    audit_service
)


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