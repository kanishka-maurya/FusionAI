from .server import mcp

from .ethics_validator import ethics_service


@mcp.tool()
async def validate_semantic_ethics(
    subtrees: list
):

    return await ethics_service.process(
        subtrees
    )