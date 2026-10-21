from app.services.embedding_service import embed_texts
from app.rag.vector_store import query_vector as qdrant_query

async def retrieve(query: str, top_k: int = 5) -> list[dict]:
    embedding = (await embed_texts([query]))[0]
    points = qdrant_query(embedding, top_k)
    return [
        {
            "score": p.get("score", 0.0),
            "text": p.get("text", ""),
            "source": p.get("source", ""),
            "page": p.get("page"),
        }
        for p in points
    ]
