from itertools import combinations
from collections import defaultdict


class CoOccurrenceAnalyzer:

    def analyze(self, subtrees):

        counts = defaultdict(int)
        totals = defaultdict(int)

        for subtree in subtrees:

            nodes = subtree["subtree"]

            for node in nodes:

                entities = node.get(
                    "associated_entities",
                    []
                )

                pairs = combinations(
                    sorted(set(entities)),
                    2
                )

                for pair in pairs:

                    counts[pair] += 1

                for pair in counts.keys():
                    totals[pair] += 1

        result = {}
        for pair, count in counts.items():

            result[str(pair)] = {
                "co_occurrence_count": count,
                "total": totals[pair]
            }

        return result


cooccurrence_analyzer = CoOccurrenceAnalyzer()