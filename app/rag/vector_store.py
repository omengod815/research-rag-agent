import os
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient, models

COLLECTION = "research_chunks"
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()

if QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL)
else:
    client = QdrantClient(path=str(Path("data/qdrant")))


def ensure_collection(vector_size: int) -> None:
    names = {c.name for c in client.get_collections().collections}
    if COLLECTION not in names:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def _make_filter(
    user_id: str | None = None,
    document_id: str | None = None,
    fingerprint: str | None = None,
):
    conditions = []
    if user_id is not None:
        conditions.append(models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)))
    if document_id is not None:
        conditions.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)))
    if fingerprint is not None:
        conditions.append(models.FieldCondition(key="document_fingerprint", match=models.MatchValue(value=fingerprint)))
    return models.Filter(must=conditions) if conditions else None


def upsert_chunks(
    chunks,
    vectors: list[list[float]],
    document_id: str,
    fingerprint: str,
    user_id: str = "local-demo",
) -> None:
    if not chunks:
        raise ValueError("没有可入库的 Chunk。请检查 PDF 是否含可提取文本。")
    if len(chunks) != len(vectors):
        raise ValueError(f"Chunk 数量 {len(chunks)} 与向量数量 {len(vectors)} 不一致。")

    ensure_collection(len(vectors[0]))
    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            **chunk.metadata,
            "text": chunk.page_content,
            "document_id": document_id,
            "document_fingerprint": fingerprint,
            "user_id": user_id,
        }
        points.append(models.PointStruct(id=str(uuid4()), vector=vector, payload=payload))
    client.upsert(collection_name=COLLECTION, points=points, wait=True)


def query_vector(
    vector: list[float],
    top_k: int = 5,
    user_id: str = "local-demo",
    document_id: str | None = None,
) -> list[dict]:
    query_filter = _make_filter(user_id=user_id, document_id=document_id)
    result = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=query_filter,
        with_payload=True,
        limit=top_k,
    ).points
    hits = []
    for point in result:
        payload = point.payload or {}
        hits.append({**payload, "id": str(point.id), "score": float(point.score)})
    return hits


def has_chunks(user_id: str = "local-demo") -> bool:
    """只查询一条记录，用于判断指定用户是否有数据，不把全文载入内存。"""
    if COLLECTION not in {c.name for c in client.get_collections().collections}:
        return False
    records, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=_make_filter(user_id=user_id),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return bool(records)


def scroll_chunks(user_id: str = "local-demo") -> list[dict]:
    """读取指定用户的 Chunk；适合本地小语料，不适合数百万条数据。"""
    if COLLECTION not in {c.name for c in client.get_collections().collections}:
        return []
    output = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=_make_filter(user_id=user_id),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            output.append({**payload, "id": str(record.id)})
        if next_offset is None:
            break
        offset = next_offset
    return output


def find_document_by_fingerprint(fingerprint: str, user_id: str = "local-demo") -> str | None:
    if COLLECTION not in {c.name for c in client.get_collections().collections}:
        return None
    records, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=_make_filter(user_id=user_id, fingerprint=fingerprint),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return None
    payload = records[0].payload or {}
    return payload.get("document_id")


def delete_document_points(document_id: str, user_id: str = "local-demo") -> None:
    if COLLECTION not in {c.name for c in client.get_collections().collections}:
        return
    query_filter = _make_filter(user_id=user_id, document_id=document_id)
    client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(filter=query_filter),
        wait=True,
    )