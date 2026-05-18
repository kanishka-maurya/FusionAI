from .server import mcp

from .topic_cube import topic_cube
from .temporal_trends import temporal_trend_analyzer
from .cooccurrence import cooccurrence_analyzer


@mcp.tool()
async def analyze_temporal_trends(subtrees: list):

    cube = topic_cube.build_topic_cube(
        subtrees
    )

    trends = temporal_trend_analyzer.analyze(
        cube
    )

    return trends


@mcp.tool()
async def analyze_topic_cooccurrence(subtrees: list):

    result = cooccurrence_analyzer.analyze(
        subtrees
    )

    return result