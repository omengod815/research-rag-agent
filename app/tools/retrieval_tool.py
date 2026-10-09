from langchain_core.tools import tool

from app.rag.retriever import retrieve


@tool
async def search_papers(query: str) -> str:
    """检索已入库科研论文，返回带 source/page 的相关证据。"""
    hits = await retrieve(query, top_k=5)
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(
            f"[{i}] {h['source']} p.{h['page']} score={h['score']:.4f}\n"
            f"{h['text']}"
        )
    return "\n\n".join(parts)