from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = "deep_research"


class ScoreDetails(BaseModel):
    hybrid_score: float = 0.0
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    keyword_score: float = 0.0
    entity_score: float = 0.0
    freshness_score: float = 0.0


class ProvenanceNode(BaseModel):
    node_id: str
    type: Optional[str] = None
    parent_id: Optional[str] = None
    child_ids: List[str] = []
    entities: List[str] = []
    summary_hash: str
    content_hash: str


class RetrievedContext(BaseModel):
    query: str
    summary: str
    key_points: List[Any] = []
    provenance: List[Dict[str, Any]] = []
    score_details: Dict[str, Any] = {}


class RetrievalQuality(BaseModel):
    query_coverage: float = 0.0
    expanded_query_count: int = 0
    retrieved_subtree_count: int = 0
    unique_root_count: int = 0
    redundancy_ratio: float = 0.0
    fallback_count: int = 0
    average_relevance_score: float = 0.0
