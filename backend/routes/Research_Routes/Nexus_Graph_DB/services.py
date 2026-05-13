import math
import numpy as np
from typing import List, Dict, Any, Set

from .config import settings, embedding_service
from .database import db

from .merger import merge_nodes_with_groq


class EngineServices:

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        arr1 = np.array(v1, dtype=np.float32)
        arr2 = np.array(v2, dtype=np.float32)

        dot = np.dot(arr1, arr2)

        n1 = np.linalg.norm(arr1)
        n2 = np.linalg.norm(arr2)

        if n1 == 0 or n2 == 0:
            return 0.0

        return float(dot / (n1 * n2))

    @staticmethod
    def normalize_embedding(vec):
        arr = np.array(vec, dtype=np.float32)

        norm = np.linalg.norm(arr)

        if norm == 0:
            return arr.tolist()

        return (arr / norm).tolist()

    @staticmethod
    def calculate_dynamic_alpha(cluster_size: int) -> float:
        return min(
            0.92,
            max(
                0.55,
                0.7 + (math.log(cluster_size + 1) / 20)
            )
        )

    @classmethod
    def calculate_threshold(cls, cluster_size: int) -> float:

        if cluster_size <= 1:
            return 1.0

        dynamic_alpha = cls.calculate_dynamic_alpha(cluster_size)

        tau = 1.0 - math.sqrt(
            (2.0 * math.log(cluster_size)) /
            settings.EMBEDDING_DIM
        ) * (1.0 - dynamic_alpha)

        return tau

    async def ingest_document(
        self,
        chunk_id: str,
        content: str,
        summary: str,
        key_points: List[str],
        entities: List[str]
    ) -> str:

        leaf_id = db.generate_id("leaf")

        raw_leaf_vector = embedding_service.generate_query_embedding(content)

        leaf_embed = self.normalize_embedding(
            raw_leaf_vector.tolist()
        )

        leaf_node = {
            "node_id": leaf_id,
            "type": "leaf",
            "node_embedding": leaf_embed,
            "summary": summary,
            "actual_content": content,
            "key_points": [],
            "associated_entities": entities,
            "parent_id": None,
            "child_ids": []
        }

        for idx, kp in enumerate(key_points):

            kp_vector = embedding_service.generate_query_embedding(kp)

            leaf_node["key_points"].append({
                "point_id": f"kp_{leaf_id}_{idx}",
                "text": kp,
                "kp_embedding": self.normalize_embedding(
                    kp_vector.tolist()
                )
            })

        db.save_node(leaf_node)

        for entity in entities:
            await self._process_cluster_merge(
                leaf_id,
                entity,
                depth=0
            )

        return leaf_id

    async def _process_cluster_merge(
        self,
        node_id: str,
        entity: str,
        depth: int = 0
    ):

        if depth > 10:
            return

        active_roots = db.get_active_roots(entity)

        if not active_roots:
            db.add_to_index(entity, node_id)
            return

        incoming_node = db.get_node(node_id)

        if not incoming_node:
            return

        incoming_vector = incoming_node["node_embedding"]

        tau = self.calculate_threshold(
            len(active_roots)
        )

        best_score = -1.0
        best_root_id = None

        for root_id in active_roots:

            if root_id == node_id:
                continue

            root_node = db.get_node(root_id)

            if not root_node:
                continue

            score = self.cosine_similarity(
                incoming_vector,
                root_node["node_embedding"]
            )

            if score > best_score:
                best_score = score
                best_root_id = root_id

        if best_score > tau and best_root_id:

            parent_id = db.generate_id("parent")

            node1 = db.get_node(best_root_id)
            node2 = db.get_node(node_id)

            merged_summary, merged_key_points = await merge_nodes_with_groq(
                node1,
                node2
            )

            parent_vector = embedding_service.generate_query_embedding(
                merged_summary
            )

            parent_vector = self.normalize_embedding(
                parent_vector.tolist()
            )

            deduped_key_points = []

            existing_vectors = []

            for idx, kp_text in enumerate(merged_key_points):

                kp_vector = embedding_service.generate_query_embedding(
                    kp_text
                )

                kp_vector = self.normalize_embedding(
                    kp_vector.tolist()
                )

                should_add = True

                for ev in existing_vectors:

                    sim = self.cosine_similarity(
                        kp_vector,
                        ev
                    )

                    if sim > 0.92:
                        should_add = False
                        break

                if should_add:

                    existing_vectors.append(kp_vector)

                    deduped_key_points.append({
                        "point_id": f"kp_{parent_id}_{idx}",
                        "text": kp_text,
                        "kp_embedding": kp_vector
                    })

            parent_node = {
                "node_id": parent_id,
                "type": "parent",
                "node_embedding": parent_vector,
                "summary": merged_summary,
                "actual_content": None,
                "key_points": deduped_key_points,
                "associated_entities": [entity],
                "parent_id": None,
                "child_ids": [
                    best_root_id,
                    node_id
                ]
            }

            db.save_node(parent_node)

            db.get_node(best_root_id)["parent_id"] = parent_id
            db.get_node(node_id)["parent_id"] = parent_id

            db.remove_from_index(entity, best_root_id)
            db.remove_from_index(entity, node_id)

            db.add_to_index(entity, parent_id)

            await self._process_cluster_merge(
                parent_id,
                entity,
                depth + 1
            )

        else:
            db.add_to_index(entity, node_id)

    def fetch_subtree(self, root_id: str):

        collected = []

        queue = [root_id]

        while queue:

            curr_id = queue.pop(0)

            node = db.get_node(curr_id)

            if not node:
                continue

            collected.append(node)

            for child in node["child_ids"]:
                queue.append(child)

        return collected

    def query_index(
        self,
        query_text: str,
        query_entities: List[str],
        top_k: int = 2
    ):

        query_vector = embedding_service.generate_query_embedding(
            query_text
        )

        query_vector = self.normalize_embedding(
            query_vector.tolist()
        )

        target_roots = set()

        for entity in query_entities:
            target_roots.update(
                db.get_active_roots(entity)
            )

        fallback_used = False

        if not target_roots:
            target_roots = db.global_active_roots
            fallback_used = True

        scored = []

        for root_id in target_roots:

            node = db.get_node(root_id)

            if not node:
                continue

            score = self.cosine_similarity(
                query_vector,
                node["node_embedding"]
            )

            scored.append(
                (root_id, score)
            )

        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )

        selected = scored[:top_k]

        payload = []

        for root_id, score in selected:

            subtree = self.fetch_subtree(root_id)

            payload.append({
                "entry_root_id": root_id,
                "relevance_score": score,
                "fallback_triggered": fallback_used,
                "graph_nodes": subtree
            })

        return payload


engine_services = EngineServices()