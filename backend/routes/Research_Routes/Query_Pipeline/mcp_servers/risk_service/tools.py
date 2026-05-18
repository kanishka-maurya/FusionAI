from .server import mcp

from .risk_scoring import risk_service


@mcp.tool()
async def analyze_cross_cluster_risk(
    parent_nodes: list
):

    return await risk_service.process(
        parent_nodes
    )