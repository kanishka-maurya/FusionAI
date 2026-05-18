class TopicDepthAnalyzer:

    def calculate_depth(self, subtree_bundle):

        topic_depths = {}

        subtree = subtree_bundle["subtree"]

        for node in subtree:

            entities = node.get(
                "associated_entities",
                []
            )

            summary = (
                node.get("summary", "")
            ).lower()

            for entity in entities:

                entity_lower = entity.lower()

                if entity_lower in summary:

                    topic_depths[entity] = (
                        topic_depths.get(entity, 0) + 1
                    )

        return topic_depths


topic_depth_analyzer = TopicDepthAnalyzer()