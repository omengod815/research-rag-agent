import asyncio

import pytest

from app.rag.langchain_basics import explain_chain, route_chain


@pytest.mark.integration
def test_langchain_explain():
    result = asyncio.run(explain_chain.ainvoke({"topic": "RAG"}))
    print("LangChain回答:", result)
    assert result.strip()


@pytest.mark.integration
def test_langchain_structured_output():
    result = asyncio.run(
        route_chain.ainvoke({"question": "这篇论文使用了什么数据集？"})
    )
    print("route:", result.route)
    print("reason:", result.reason)
    assert result.route in {"retrieval", "data", "general"}
