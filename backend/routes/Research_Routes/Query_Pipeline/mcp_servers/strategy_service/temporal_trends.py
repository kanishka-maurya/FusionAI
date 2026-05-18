from collections import defaultdict


class TemporalTrendAnalyzer:

    def analyze(self, cube):

        trends = defaultdict(list)

        for topic, rows in cube.items():

            sorted_rows = sorted(
                rows,
                key=lambda x: x["created_at"] or ""
            )

            trends[topic] = [
                row["created_at"]
                for row in sorted_rows
            ]

        return trends


temporal_trend_analyzer = TemporalTrendAnalyzer()