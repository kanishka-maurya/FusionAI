from itertools import combinations


class PairBuilder:

    def build_pairs(self, parent_nodes):

        return list(
            combinations(parent_nodes, 2)
        )


pair_builder = PairBuilder()