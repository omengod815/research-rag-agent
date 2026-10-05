from app.services.embedding_service import embed_texts
from app.rag.vector_store import query_vector as qdrant_query

async def retrieve(query: str, top_k: int = 5) -> list[dict]:
    embedding = (await embed_texts([query]))[0]
    points = qdrant_query(embedding, top_k)
    return [
        {
            "score": float(p.score),
            "text": (p.payload or {}).get("text", ""),
            "source": (p.payload or {}).get("source", ""),
            "page": (p.payload or {}).get("page"),
        }
        for p in points
    ]