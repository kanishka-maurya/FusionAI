from collections import defaultdict


class TopicCube:

    def build_topic_cube(self, subtrees):

        cube = defaultdict(list)

        for subtree in subtrees:

            nodes = subtree["subtree"]

            for node in nodes:

                entities = node.get(
                    "associated_entities",
                    []
                )

                created_at = node.get(
                    "created_at"
                )

                for entity in entities:

                    cube[entity].append({
                        "node_id": node["node_id"],
                        "created_at": created_at,
                        "summary": node["summary"]
                    })

        return cube
    
topic_cube = TopicCube()