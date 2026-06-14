import hashlib
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence


SECTION_RE = re.compile(
    r"(?im)^(#{1,6}\s+.+|[A-Z][A-Z0-9\s:,-]{8,}|(?:abstract|introduction|methodology|methods|results|discussion|conclusion|references)\s*)$"
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}")


def normalize_entity(entity: str) -> str:
    return re.sub(r"\s+", " ", str(entity).strip()).lower()


def tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if len(token) > 2
    ]


def lexical_overlap(query: str, text: str) -> float:
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(text))

    if not query_terms or not text_terms:
        return 0.0

    return len(query_terms & text_terms) / len(query_terms)


def keyword_density(query: str, text: str) -> float:
    query_terms = tokenize(query)
    if not query_terms:
        return 0.0

    counts = Counter(tokenize(text))
    hits = sum(counts.get(term, 0) for term in query_terms)
    return min(1.0, hits / max(1, len(query_terms)))


def parse_datetime(value: Any):
    if not value:
        return None

    if isinstance(value, datetime):
        return value

    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def freshness_score(value: Any, half_life_days: int = 30) -> float:
    created_at = parse_datetime(value)
    if not created_at:
        return 0.25

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    age_days = max(
        0.0,
        (datetime.now(timezone.utc) - created_at).total_seconds() / 86400,
    )
    return math.exp(-age_days / max(1, half_life_days))


def source_authority_score(item: Dict[str, Any]) -> float:
    source = item.get("source") or item.get("source_type") or ""
    meta = item.get("meta") or {}

    if source == "github":
        stars = int(meta.get("stars") or 0)
        return min(1.0, 0.25 + math.log10(stars + 1) / 5)

    if source == "papers":
        return 0.9

    if source == "news":
        source_name = str(meta.get("source_name") or "").lower()
        if source_name:
            return 0.65
        return 0.45

    return 0.5


def novelty_score(text: str, previous_texts: Sequence[str]) -> float:
    if not previous_texts:
        return 1.0

    current_terms = set(tokenize(text))
    if not current_terms:
        return 0.0

    max_overlap = 0.0
    for previous in previous_texts:
        previous_terms = set(tokenize(previous))
        if previous_terms:
            max_overlap = max(
                max_overlap,
                len(current_terms & previous_terms) / len(current_terms),
            )

    return max(0.0, 1.0 - max_overlap)


def stable_hash(*parts: Any, length: int = 12) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def keypoint_texts(node: Dict[str, Any]) -> List[str]:
    texts = []
    for keypoint in node.get("key_points", []) or []:
        if isinstance(keypoint, dict):
            text = keypoint.get("text")
        else:
            text = str(keypoint)
        if text:
            texts.append(str(text))
    return texts


def node_entities(node: Dict[str, Any]) -> List[str]:
    entities = (
        node.get("associated_entities")
        or node.get("entities")
        or []
    )
    return [str(entity) for entity in entities if str(entity).strip()]


def node_text(node: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(node.get("summary") or ""),
            str(node.get("actual_content") or ""),
            " ".join(keypoint_texts(node)),
            " ".join(node_entities(node)),
        ]
    )


def build_provenance(node: Dict[str, Any]) -> Dict[str, Any]:
    node_id = node.get("node_id", "")
    return {
        "node_id": node_id,
        "type": node.get("type"),
        "parent_id": node.get("parent_id"),
        "child_ids": node.get("child_ids") or [],
        "entities": node_entities(node),
        "summary_hash": stable_hash(node.get("summary", "")),
        "content_hash": stable_hash(node.get("actual_content", "")),
    }


def semantic_chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 180,
) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    section_markers = SECTION_RE.split(text)
    blocks = []

    if len(section_markers) > 1:
        for marker, body in zip(section_markers[1::2], section_markers[2::2]):
            block = f"{marker.strip()}. {body.strip()}"
            if block.strip():
                blocks.append(block)
    else:
        blocks = re.split(r"\n{2,}", text)

    sentences = []
    for block in blocks or [text]:
        sentences.extend(
            sentence.strip()
            for sentence in SENTENCE_RE.split(block)
            if sentence.strip()
        )

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current and current_len + sentence_len > chunk_size:
            chunk = " ".join(current).strip()
            chunks.append(chunk)

            overlap_text = chunk[-overlap:] if overlap > 0 else ""
            current = [overlap_text, sentence] if overlap_text else [sentence]
            current_len = sum(len(part) for part in current)
        else:
            current.append(sentence)
            current_len += sentence_len

    if current:
        chunks.append(" ".join(current).strip())

    return [
        chunk
        for chunk in chunks
        if len(chunk.strip()) > 40
    ] or [text[:chunk_size]]


def hybrid_relevance_score(
    query: str,
    query_entities: Iterable[str],
    node: Dict[str, Any],
    semantic_score: float,
) -> Dict[str, float]:
    text = node_text(node)
    node_entity_set = {
        normalize_entity(entity)
        for entity in node_entities(node)
    }
    query_entity_set = {
        normalize_entity(entity)
        for entity in query_entities
    }

    entity_score = (
        len(node_entity_set & query_entity_set) / len(query_entity_set)
        if query_entity_set
        else 0.0
    )
    lexical_score = lexical_overlap(query, text)
    keyword_score = keyword_density(query, text)
    fresh_score = freshness_score(node.get("created_at"))

    hybrid = (
        0.58 * max(0.0, semantic_score)
        + 0.17 * lexical_score
        + 0.10 * keyword_score
        + 0.10 * entity_score
        + 0.05 * fresh_score
    )

    return {
        "hybrid_score": round(hybrid, 6),
        "semantic_score": round(float(semantic_score), 6),
        "lexical_score": round(lexical_score, 6),
        "keyword_score": round(keyword_score, 6),
        "entity_score": round(entity_score, 6),
        "freshness_score": round(fresh_score, 6),
    }


def retrieval_quality_report(
    query: str,
    expanded_queries: Sequence[str],
    subtrees: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    root_ids = [
        subtree.get("root_node", {}).get("node_id")
        for subtree in subtrees
        if subtree.get("root_node")
    ]
    unique_roots = set(root_id for root_id in root_ids if root_id)
    fallback_count = sum(
        1 for subtree in subtrees if subtree.get("fallback_used")
    )
    retrieved_text = " ".join(
        node_text(subtree.get("root_node", {}))
        for subtree in subtrees
    )

    return {
        "query_coverage": round(
            lexical_overlap(query, retrieved_text),
            6,
        ),
        "expanded_query_count": len(expanded_queries),
        "retrieved_subtree_count": len(subtrees),
        "unique_root_count": len(unique_roots),
        "redundancy_ratio": round(
            1.0 - (len(unique_roots) / max(1, len(root_ids))),
            6,
        ),
        "fallback_count": fallback_count,
        "average_relevance_score": round(
            sum(
                float(subtree.get("relevance_score") or 0.0)
                for subtree in subtrees
            ) / max(1, len(subtrees)),
            6,
        ),
    }


def detect_contradictions(
    retrieved_contexts: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    contradiction_markers = [
        ("increase", "decrease"),
        ("improve", "worsen"),
        ("safe", "unsafe"),
        ("open-source", "closed-source"),
        ("faster", "slower"),
        ("higher", "lower"),
        ("supports", "does not support"),
    ]
    findings = []

    for i, left in enumerate(retrieved_contexts):
        left_text = str(left.get("summary") or "").lower()
        for right in retrieved_contexts[i + 1:]:
            right_text = str(right.get("summary") or "").lower()
            for positive, negative in contradiction_markers:
                left_has_pair = positive in left_text and negative in right_text
                right_has_pair = negative in left_text and positive in right_text
                if left_has_pair or right_has_pair:
                    findings.append({
                        "left_query": left.get("query", ""),
                        "right_query": right.get("query", ""),
                        "marker_pair": [positive, negative],
                        "confidence": "low",
                        "reason": "Opposing directional language appeared in retrieved summaries.",
                    })
                    break

    return findings[:5]
