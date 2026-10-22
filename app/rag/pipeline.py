from app.rag.retriever import retrieve
from app.services.llm_service import chat_once


async def rag_answer(question: str, top_k: int = 5, user_id: str = "local-demo") -> dict:
    hits = await retrieve(question, top_k=top_k, user_id=user_id)
    context = "\n\n".join(
        f"[{index}] source={hit.get('source')} page={hit.get('page')}\n{hit.get('text', '')}"
        for index, hit in enumerate(hits, start=1)
    )
    prompt = f"""
你是科研论文问答助手。只能根据检索资料作答；资料不足时明确说“当前资料不足以回答”。
回答中的 [1]、[2] 等编号必须对应下面的资料。
问题：{question}
检索资料：
{context or '没有检索到相关资料。'}
""".strip()
    answer = await chat_once(prompt)
    return {"answer": answer, "sources": hits}