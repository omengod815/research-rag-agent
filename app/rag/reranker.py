import logging
import os
from functools import lru_cache

import jieba

logger = logging.getLogger(__name__)


def _tokens(text: str) -> set[str]:
    return {item.strip().lower() for item in jieba.lcut(text) if item.strip()}


def keyword_overlap_rerank(query: str, hits: list[dict]) -> list[dict]:
    query_tokens = _tokens(query)
    rescored = []
    for hit in hits:
        item = dict(hit)
        overlap = len(query_tokens & _tokens(item.get("text", "")))
        item["rerank_score"] = float(item.get("rrf_score", item.get("score", 0.0))) + 0.001 * overlap
        rescored.append(item)
    return sorted(rescored, key=lambda item: item["rerank_score"], reverse=True)


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder
    model_name = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
    return CrossEncoder(model_name, max_length=512)


def rerank_hits(query: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return []
    if os.getenv("ENABLE_CROSS_ENCODER", "0") != "1":
        return keyword_overlap_rerank(query, hits)

    try:
        model = _get_cross_encoder()
        pairs = [(query, item.get("text", "")) for item in hits]
        scores = model.predict(pairs)
        output = []
        for item, score in zip(hits, scores):
            copied = dict(item)
            copied["rerank_score"] = float(score)
            output.append(copied)
        return sorted(output, key=lambda item: item["rerank_score"], reverse=True)
    except Exception:
        logger.exception("Cross-Encoder 加载或推理失败，退回 keyword overlap baseline")
        return keyword_overlap_rerank(query, hits)