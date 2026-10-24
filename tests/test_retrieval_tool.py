import pytest
from app.tools.retrieval_tool import search_papers


@pytest.mark.asyncio
async def test_search_papers_returns_formatted_evidence():
    """验证 Retrieval Tool 能正确调用并返回格式化证据"""
    result = await search_papers.ainvoke({"query": "大语言模型"})

    # 基础类型与内容校验
    assert isinstance(result, str), "Tool 返回值必须是字符串"
    assert len(result) > 0, "检索结果为空，请检查 Qdrant 数据或 Embedding 一致性"
    assert "[1]" in result, "返回格式异常，缺少证据编号标记"


@pytest.mark.asyncio
async def test_search_papers_handles_empty_query():
    """验证空查询不会导致 Tool 崩溃"""
    result = await search_papers.ainvoke({"query": ""})
    assert isinstance(result, str), "空查询也应返回字符串而非抛出异常"


@pytest.mark.asyncio
async def test_search_papers_rejects_invalid_params():
    """验证参数 Schema 校验生效（阶段14工程化要求）"""
    with pytest.raises(Exception):
        # 传入错误的参数名，应触发 Pydantic/LangChain 校验异常
        await search_papers.ainvoke({"wrong_param": "test"})