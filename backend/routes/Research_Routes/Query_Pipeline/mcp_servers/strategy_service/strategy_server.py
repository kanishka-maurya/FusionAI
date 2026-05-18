from .topic_cube import topic_cube
from .temporal_trends import temporal_trend_analyzer
from .topic_depth import topic_depth_analyzer
from .cooccurrence import cooccurrence_analyzer


class StrategyService:

    async def process(self, subtrees):

        cube = topic_cube.build_topic_cube(
            subtrees
        )

        temporal_trends = (
            temporal_trend_analyzer.analyze(cube)
        )

        depths = []

        for subtree in subtrees:

            depths.append(
                topic_depth_analyzer.calculate_depth(
                    subtree
                )
            )

        cooccurrence = (
            cooccurrence_analyzer.analyze(
                subtrees
            )
        )
        return {
            "topic_cube": cube,
            "temporal_trends": temporal_trends,
            "topic_depths": depths,
            "cooccurrence": cooccurrence
        }


strategy_service = StrategyService()