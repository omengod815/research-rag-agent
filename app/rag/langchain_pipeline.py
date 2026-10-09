from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.rag.retriever import retrieve
from app.services.langchain_llm import get_langchain_llm


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是科研论文问答助手。只能依据给定资料回答。"
        "如果资料不足，请明确说：当前资料不足以回答。"
        "回答时使用 [1][2] 这样的编号引用证据。",
    ),
    (
        "human",
        "问题：{question}\n\n检索资料：\n{context}",
    ),
])

rag_generation_chain = (
    RAG_PROMPT
    | get_langchain_llm()
    | StrOutputParser()
)


def format_hits(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(
            f"[{i}] source={h['source']} page={h['page']}\n{h['text']}"
        )
    return "\n\n".join(parts)


async def langchain_rag_answer(question: str, top_k: int = 5) -> dict:
    # 检索仍然复用你旧 A 已经验证过的 Qwen Embedding + Qdrant
    hits = await retrieve(question, top_k=top_k)
    context = format_hits(hits)

    answer = await rag_generation_chain.ainvoke({
        "question": question,
        "context": context,
    })

    return {
        "answer": answer,
        "sources": hits,
        "engine": "langchain",
    }