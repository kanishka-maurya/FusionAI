from typing import Any, Dict, List

from backend.routes.Research_Routes.utils import (
    lexical_overlap,
    node_entities,
    node_text,
)


def graph_health_report(subtrees: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes = []
    for subtree in subtrees:
        nodes.extend(subtree.get("subtree", []))

    node_ids = [node.get("node_id") for node in nodes]
    unique_node_ids = set(node_id for node_id in node_ids if node_id)
    parent_count = sum(1 for node in nodes if node.get("type") == "parent")
    leaf_count = sum(1 for node in nodes if node.get("type") == "leaf")
    orphan_count = sum(
        1
        for node in nodes
        if node.get("type") == "leaf" and not node.get("parent_id")
    )

    entity_count = sum(len(node_entities(node)) for node in nodes)

    return {
        "node_count": len(nodes),
        "unique_node_count": len(unique_node_ids),
        "parent_count": parent_count,
        "leaf_count": leaf_count,
        "orphan_leaf_count": orphan_count,
        "average_entities_per_node": round(
            entity_count / max(1, len(nodes)),
            6,
        ),
    }


def duplicate_merge_candidates(
    nodes: List[Dict[str, Any]],
    threshold: float = 0.82,
) -> List[Dict[str, Any]]:
    candidates = []

    for index, left in enumerate(nodes):
        left_text = node_text(left)
        for right in nodes[index + 1:]:
            overlap = lexical_overlap(left_text, node_text(right))
            if overlap >= threshold:
                candidates.append({
                    "node_1": left.get("node_id"),
                    "node_2": right.get("node_id"),
                    "lexical_overlap": round(overlap, 6),
                    "recommendation": "review_for_merge",
                })

    return candidates[:20]
