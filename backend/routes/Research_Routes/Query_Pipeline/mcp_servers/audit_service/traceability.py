class Traceability:

    def trace(self, subtree):

        paths = []

        for node in subtree:

            paths.append({
                "node_id": node["node_id"],
                "parent_id": node.get("parent_id"),
                "children": node.get("child_ids", [])
            })

        return paths


traceability = Traceability()