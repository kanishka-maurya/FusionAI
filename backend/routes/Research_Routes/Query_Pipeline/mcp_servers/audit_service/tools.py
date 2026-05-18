from .server import mcp

from .audit_server import audit_service


@mcp.tool()
async def trace_graph_consistency(
    subtrees: list
):

    return await audit_service.process(
        subtrees
    )