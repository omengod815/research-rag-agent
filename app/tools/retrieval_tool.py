from langchain_core.tools import tool
from app.rag.retriever import retrieve

@tool
async def search_papers(query: str) -> str:
    """Search indexed research papers and return evidence with source/page."""
    hits = await retrieve(query, top_k=5)
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(
            f"[{i}] {h['source']} p.{h['page']} score={h['score']:.3f}\n{h['text']}"
        )
    return "\n\n".join(parts)