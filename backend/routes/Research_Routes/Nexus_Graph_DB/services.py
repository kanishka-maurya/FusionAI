import math
import logging
import numpy as np
from typing import List
from supabase import create_client, Client
import os
from dotenv import load_dotenv

from .config import settings, embedding_service
from .merger import merge_nodes_with_groq
from backend.routes.Research_Routes.utils import hybrid_relevance_score

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


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
    def calculate_dynamic_alpha(cluster_size: int):

        alpha = min(
            0.92,
            max(
                0.55,
                0.7 + (
                    math.log(cluster_size + 1) / 20
                )
            )
        )

        logging.info(
            f"[THRESHOLD] Dynamic alpha calculated: "
            f"{alpha:.4f} for cluster size {cluster_size}"
        )

        return alpha

    @classmethod
    def calculate_threshold(cls, cluster_size: int):

        if cluster_size <= 1:
            return 1.0

        dynamic_alpha = cls.calculate_dynamic_alpha(
            cluster_size
        )

        tau = 1.0 - math.sqrt(
            (
                2.0 * math.log(cluster_size)
            ) / settings.EMBEDDING_DIM
        ) * (
            1.0 - dynamic_alpha
        )

        logging.info(
            f"[THRESHOLD] Tau calculated: {tau:.4f}"
        )

        return tau

    async def save_node(self, node):

        logging.info(
            f"[DB] Saving node: {node['node_id']} "
            f"| Type: {node['type']}"
        )

        supabase.table("graph_nodes").upsert({
            "node_id": node["node_id"],
            "type": node["type"],
            "node_embedding": node["node_embedding"],
            "summary": node["summary"],
            "actual_content": node["actual_content"],
            "key_points": node["key_points"],
            "associated_entities": node["associated_entities"],
            "parent_id": node["parent_id"],
            "child_ids": node["child_ids"]
        }).execute()

        logging.info(
            f"[DB] Node persisted successfully: "
            f"{node['node_id']}"
        )

    async def get_node(self, node_id):

        response = supabase.table(
            "graph_nodes"
        ).select("*").eq(
            "node_id",
            node_id
        ).execute()

        data = response.data

        if not data:

            logging.warning(
                f"[DB] Node not found: {node_id}"
            )

            return None

        return data[0]

    async def add_to_index(self, entity, node_id):

        logging.info(
            f"[INDEX] Adding entity mapping: "
            f"{entity} -> {node_id}"
        )

        existing = supabase.table(
            "entity_index"
        ).select("*").eq(
            "entity",
            entity
        ).eq(
            "node_id",
            node_id
        ).execute()

        if not existing.data:

            supabase.table(
                "entity_index"
            ).insert({
                "entity": entity,
                "node_id": node_id
            }).execute()

            logging.info(
                f"[INDEX] Mapping inserted successfully"
            )

    async def remove_from_index(self, entity, node_id):

        logging.info(
            f"[INDEX] Removing mapping: "
            f"{entity} -> {node_id}"
        )

        supabase.table(
            "entity_index"
        ).delete().eq(
            "entity",
            entity
        ).eq(
            "node_id",
            node_id
        ).execute()

    async def get_active_roots(self, entity):

        response = supabase.table(
            "entity_index"
        ).select(
            "node_id"
        ).eq(
            "entity",
            entity
        ).execute()

        roots = [
            row["node_id"]
            for row in response.data
        ]

        logging.info(
            f"[INDEX] Active roots for entity "
            f"'{entity}': {roots}"
        )

        return roots

    async def fetch_all_global_roots(self):

        response = supabase.table(
            "graph_nodes"
        ).select(
            "node_id"
        ).is_(
            "parent_id",
            "null"
        ).execute()

        roots = [
            row["node_id"]
            for row in response.data
        ]

        logging.info(
            f"[GLOBAL ROOTS] Total roots fetched: "
            f"{len(roots)}"
        )

        return roots

    async def ingest_document(
        self,
        chunk_id,
        content,
        summary,
        key_points,
        entities,
        metadata=None
    ):

        logging.info(
            f"[INGESTION] Starting ingestion "
            f"for chunk: {chunk_id}"
        )

        leaf_id = f"leaf_{chunk_id}"

        raw_leaf_vector = (
            embedding_service.generate_query_embedding(
                content
            )
        )

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
            "child_ids": [],
            "_metadata": metadata or {}
        }

        logging.info(
            f"[INGESTION] Generating embeddings "
            f"for {len(key_points)} key points"
        )

        for idx, kp in enumerate(key_points):

            kp_vector = (
                embedding_service.generate_query_embedding(
                    kp
                )
            )

            leaf_node["key_points"].append({
                "point_id": f"kp_{leaf_id}_{idx}",
                "text": kp,
                "kp_embedding": self.normalize_embedding(
                    kp_vector.tolist()
                )
            })

        await self.save_node(leaf_node)

        logging.info(
            f"[INGESTION] Leaf node inserted: "
            f"{leaf_id}"
        )

        for entity in entities:

            logging.info(
                f"[INGESTION] Processing entity "
                f"cluster merge for entity: {entity}"
            )

            await self._process_cluster_merge(
                leaf_id,
                entity,
                depth=0
            )

        logging.info(
            f"[INGESTION] Completed ingestion "
            f"for {leaf_id}"
        )

        return leaf_id

    async def _process_cluster_merge(
        self,
        node_id,
        entity,
        depth=0
    ):

        logging.info(
            f"[MERGE] Checking merge for node "
            f"{node_id} under entity '{entity}' "
            f"| Depth: {depth}"
        )

        if depth > 10:

            logging.warning(
                f"[MERGE] Max recursion depth reached"
            )

            return

        active_roots = await self.get_active_roots(
            entity
        )

        if not active_roots:

            logging.info(
                f"[MERGE] No active roots found. "
                f"Adding {node_id} as root."
            )

            await self.add_to_index(
                entity,
                node_id
            )

            return

        incoming_node = await self.get_node(
            node_id
        )

        if not incoming_node:
            return

        incoming_vector = incoming_node[
            "node_embedding"
        ]

        tau = self.calculate_threshold(
            len(active_roots)
        )

        best_score = -1.0
        best_root_id = None

        for root_id in active_roots:

            if root_id == node_id:
                continue

            root_node = await self.get_node(
                root_id
            )

            if not root_node:
                continue

            score = self.cosine_similarity(
                incoming_vector,
                root_node["node_embedding"]
            )

            logging.info(
                f"[SIMILARITY] "
                f"{node_id} <-> {root_id} "
                f"= {score:.4f}"
            )

            if score > best_score:

                best_score = score
                best_root_id = root_id

        logging.info(
            f"[MERGE] Best root: {best_root_id} "
            f"| Score: {best_score:.4f} "
            f"| Tau: {tau:.4f}"
        )

        if best_score > tau and best_root_id:

            logging.info(
                f"[MERGE] MERGE TRIGGERED between "
                f"{node_id} and {best_root_id}"
            )

            node1 = await self.get_node(
                best_root_id
            )

            node2 = await self.get_node(
                node_id
            )

            parent_id = (
                f"parent_{node_id}_{best_root_id}"
            )

            logging.info(
                f"[MERGE] Calling Groq merger..."
            )

            merged_summary, merged_key_points = (
                await merge_nodes_with_groq(
                    node1,
                    node2
                )
            )

            logging.info(
                f"[MERGE] Groq merge completed"
            )

            parent_vector = (
                embedding_service.generate_query_embedding(
                    merged_summary
                )
            )

            parent_vector = self.normalize_embedding(
                parent_vector.tolist()
            )

            kp_objects = []

            for idx, kp in enumerate(
                merged_key_points
            ):

                kp_vec = (
                    embedding_service.generate_query_embedding(
                        kp
                    )
                )

                kp_objects.append({
                    "point_id": f"kp_{parent_id}_{idx}",
                    "text": kp,
                    "kp_embedding": self.normalize_embedding(
                        kp_vec.tolist()
                    )
                })

            parent_node = {
                "node_id": parent_id,
                "type": "parent",
                "node_embedding": parent_vector,
                "summary": merged_summary,
                "actual_content": None,
                "key_points": kp_objects,
                "associated_entities": [entity],
                "parent_id": None,
                "child_ids": [
                    best_root_id,
                    node_id
                ],
                "_metadata": {
                    "merge_score": best_score,
                    "merge_threshold": tau
                }
            }

            await self.save_node(parent_node)

            logging.info(
                f"[MERGE] Parent node created: "
                f"{parent_id}"
            )

            node1["parent_id"] = parent_id
            node2["parent_id"] = parent_id

            await self.save_node(node1)
            await self.save_node(node2)

            logging.info(
                f"[MERGE] Updated child parent pointers"
            )

            await self.remove_from_index(
                entity,
                best_root_id
            )

            await self.remove_from_index(
                entity,
                node_id
            )

            await self.add_to_index(
                entity,
                parent_id
            )

            logging.info(
                f"[MERGE] Parent node promoted "
                f"as active root"
            )

            await self._process_cluster_merge(
                parent_id,
                entity,
                depth + 1
            )

        else:

            logging.info(
                f"[MERGE] Merge rejected. "
                f"Node added as separate root."
            )

            await self.add_to_index(
                entity,
                node_id
            )

    async def fetch_subtree(self, root_id):

        logging.info(
            f"[SUBTREE] Fetching subtree "
            f"from root: {root_id}"
        )

        collected = []

        queue = [root_id]

        while queue:

            curr_id = queue.pop(0)

            node = await self.get_node(
                curr_id
            )

            if not node:
                continue

            collected.append(node)

            for child in node["child_ids"]:
                queue.append(child)

        logging.info(
            f"[SUBTREE] Total nodes fetched: "
            f"{len(collected)}"
        )

        return collected

    async def query_index(
        self,
        query_text,
        query_entities,
        top_k=2
    ):

        logging.info(
            f"[QUERY] Processing query: "
            f"{query_text}"
        )

        query_vector = (
            embedding_service.generate_query_embedding(
                query_text
            )
        )

        query_vector = self.normalize_embedding(
            query_vector.tolist()
        )

        target_roots = set()

        for entity in query_entities:

            roots = await self.get_active_roots(
                entity
            )

            target_roots.update(roots)

        fallback_used = False

        if not target_roots:

            logging.warning(
                "[QUERY] Fallback triggered"
            )

            roots = await self.fetch_all_global_roots()

            target_roots.update(roots)

            fallback_used = True

        scored = []

        for root_id in target_roots:

            node = await self.get_node(
                root_id
            )

            if not node:
                continue

            score = self.cosine_similarity(
                query_vector,
                node["node_embedding"]
            )
            score_details = hybrid_relevance_score(
                query_text,
                query_entities,
                node,
                score
            )

            logging.info(
                f"[QUERY SCORE] "
                f"{root_id} -> {score_details['hybrid_score']:.4f}"
            )

            scored.append(
                (root_id, score_details)
            )

        scored.sort(
            key=lambda x: x[1]["hybrid_score"],
            reverse=True
        )

        selected = scored[:top_k]

        payload = []

        for root_id, score_details in selected:

            subtree = await self.fetch_subtree(
                root_id
            )

            payload.append({
                "entry_root_id": root_id,
                "relevance_score": score_details["hybrid_score"],
                "score_details": score_details,
                "fallback_triggered": fallback_used,
                "graph_nodes": subtree
            })

        logging.info(
            f"[QUERY] Query completed successfully"
        )

        return payload


engine_services = EngineServices()
