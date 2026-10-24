from collections import defaultdict

import jieba
from rank_bm25 import BM25Okapi

from app.rag.reranker import rerank_hits
from app.rag.vector_store import has_chunks, query_vector, scroll_chunks
from app.services.embedding_service import embed_texts


def _tokenize(text: str) -> list[str]:
    return [token.strip().lower() for token in jieba.lcut(text) if token.strip()]


async def retrieve(
    query: str,
    top_k: int = 5,
    user_id: str = "local-demo",
    mode: str = "hybrid",
) -> list[dict]:
    query = query.strip()
    if not query:
        return []

    if mode not in {"dense", "hybrid"}:
        raise ValueError("mode 只支持 dense 或 hybrid")
    if mode == "dense" and not has_chunks(user_id=user_id):
        return []

    all_chunks = scroll_chunks(user_id=user_id) if mode == "hybrid" else []
    if mode == "hybrid" and not all_chunks:
        return []

    query_vector_value = (await embed_texts([query]))[0]
    dense_hits = query_vector(
        query_vector_value,
        top_k=max(top_k * 4, 20),
        user_id=user_id,
    )
    if mode == "dense":
        return dense_hits[:top_k]

    tokenized_corpus = [_tokenize(item.get("text", "")) for item in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(all_chunks)), key=lambda idx: bm25_scores[idx], reverse=True)
    lexical_hits = [all_chunks[idx] for idx in ranked_indices[:max(top_k * 4, 20)]]

    # Reciprocal Rank Fusion：只融合排名，不直接把 Dense 分数与 BM25 原始分数相加。
    rrf_scores: dict[str, float] = defaultdict(float)
    hit_by_id: dict[str, dict] = {}
    rrf_constant = 60
    for ranking in (dense_hits, lexical_hits):
        for rank, hit in enumerate(ranking, start=1):
            hit_id = str(hit["id"])
            rrf_scores[hit_id] += 1.0 / (rrf_constant + rank)
            hit_by_id[hit_id] = {**hit_by_id.get(hit_id, {}), **hit}

    fused = []
    for hit_id, item in hit_by_id.items():
        fused.append({**item, "rrf_score": rrf_scores[hit_id]})
    fused.sort(key=lambda item: item["rrf_score"], reverse=True)

    candidates = fused[:max(top_k * 4, 20)]
    return rerank_hits(query, candidates)[:top_k]