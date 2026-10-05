from pathlib import Path
from uuid import uuid4
from qdrant_client import QdrantClient, models

COLLECTION = "research_chunks"
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


def upsert_chunks(chunks, vectors: list[list[float]]) -> None:
    ensure_collection(len(vectors[0]))
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "text": chunk.page_content,
                    **chunk.metadata,
                },
            )
        )
    client.upsert(collection_name=COLLECTION, points=points, wait=True)


def query_vector(vector: list[float], top_k: int = 5):
    return client.query_points(
        collection_name=COLLECTION,
        query=vector,
        with_payload=True,
        limit=top_k,
    ).points