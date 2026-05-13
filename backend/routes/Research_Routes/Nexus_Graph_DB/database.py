from typing import Dict, Any, Set, List
from datetime import datetime

from .config import settings

class GraphStorage:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.inverted_index: Dict[str, Set[str]] = {}
        self.global_active_roots: Set[str] = set()
        self.node_counter: int = 0

    def validate_node_schema(self, node: Dict[str, Any]) -> bool:
        """
        Strict structural validator verifying compliance of both Leaf 
        and Parent representations before committing updates to storage.
        """
        required_keys = {
            "node_id", "type", "node_embedding", "summary", 
            "key_points", "associated_entities", "parent_id", 
            "child_ids"
        }
        if not required_keys.issubset(node.keys()):
            raise ValueError(f"Node missing one or more required keys: {required_keys - node.keys()}")

        if node["type"] not in ["leaf", "parent"]:
            raise ValueError("Field 'type' must be explicitly evaluated as 'leaf' or 'parent'.")

        if node["type"] == "leaf":
            if "actual_content" not in node or not isinstance(node["actual_content"], str):
                raise ValueError("Type 'leaf' requires a valid string type in 'actual_content'.")
            if len(node["child_ids"]) > 0:
                raise ValueError("Type 'leaf' cannot possess active references inside 'child_ids'.")


        if node["type"] == "parent":
            if node.get("actual_content") is not None:
                raise ValueError("Type 'parent' must maintain 'actual_content' as None/null.")
            if len(node["child_ids"]) < 2:
                raise ValueError("Type 'parent' must link to at least 2 cluster elements in 'child_ids'.")

        for idx, kp in enumerate(node["key_points"]):
            kp_keys = {"point_id", "text", "kp_embedding"}
            if not kp_keys.issubset(kp.keys()):
                raise ValueError(f"Key point at index {idx} missing structural properties: {kp_keys - kp.keys()}")
            if not isinstance(kp["kp_embedding"], list) or len(kp["kp_embedding"]) != settings.EMBEDDING_DIM:
                raise ValueError(f"Key point vector space dimensionality anomaly at index {idx}. Must be length 768.")
        if not isinstance(node["node_embedding"], list) or len(node["node_embedding"]) != settings.EMBEDDING_DIM:
            raise ValueError("Core node vector mapping anomaly. Dimension space length must equal 768.")

        return True

    def generate_id(self, node_type: str) -> str:
        self.node_counter += 1
        return f"{node_type}_{self.node_counter:04d}"

    def save_node(self, node: Dict[str, Any]) -> None:
        self.validate_node_schema(node)
    
        if "created_at" not in node:
            node["created_at"] = datetime.utcnow().isoformat() + "Z"
            
        self.nodes[node["node_id"]] = node

    def get_node(self, node_id: str) -> Dict[str, Any]:
        return self.nodes.get(node_id)

    def add_to_index(self, entity: str, node_id: str) -> None:
        if entity not in self.inverted_index:
            self.inverted_index[entity] = set()
        self.inverted_index[entity].add(node_id)

    def remove_from_index(self, entity: str, node_id: str) -> None:
        if entity in self.inverted_index and node_id in self.inverted_index[entity]:
            self.inverted_index[entity].remove(node_id)

    def get_active_roots(self, entity: str) -> List[str]:
        return list(self.inverted_index.get(entity, set()))

db = GraphStorage()
