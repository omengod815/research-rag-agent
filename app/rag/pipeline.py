from app.rag.retriever import retrieve
from app.services.llm_service import chat_once

async def rag_answer(question: str, top_k: int = 5) -> dict:
    hits = await retrieve(question, top_k=top_k)
    context_parts = []
    for i, h in enumerate(hits, start=1):
        context_parts.append(
            f"[{i}] source={h['source']} page={h['page']}\n{h['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""
你是科研论文问答助手。只能依据给定资料回答。
如果资料不足，请明确说“当前资料不足以回答”。
回答后标注引用编号，例如 [1][3]。

问题：{question}

资料：
{context}
""".strip()

    answer = await chat_once(prompt)
    return {
        "answer": answer,
        "sources": hits,
    }