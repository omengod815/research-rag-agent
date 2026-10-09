import asyncio

from app.rag.langchain_basics import get_explain_chain, get_route_chain


def test_langchain_explain():
    chain = get_explain_chain()
    result = asyncio.run(chain.ainvoke({"topic": "RAG"}))
    print("LangChain回答:", result)
    assert result.strip()


def test_langchain_structured_output():
    chain = get_route_chain()
    result = asyncio.run(
        chain.ainvoke({"question": "这篇论文使用了什么数据集？"})
    )
    print("route:", result.route)
    print("reason:", result.reason)
    assert result.route in {"retrieval", "data", "general"}