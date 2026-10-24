import pytest
from app.agents.router import route_question

# 定义测试用例集：(问题, 期望路由)
ROUTING_TEST_CASES = [
    # ✅ retrieval: 论文/知识库查询
    ("这篇论文的数据集是什么", "retrieval"),
    ("Transformer 的自注意力机制原理", "retrieval"),
    ("RAG 系统中向量检索的优缺点", "retrieval"),
    ("帮我找一下关于 LLM 幻觉的最新研究", "retrieval"),
    ("BERT 和 RoBERTa 的区别是什么", "retrieval"),
    
    # ✅ data: CSV/数值分析
    ("帮我算 prediction 和 true 的 MAE", "data"),
    ("计算这个 CSV 文件中 accuracy 列的平均值", "data"),
    ("绘制 loss 随 epoch 变化的曲线", "data"),
    ("统计 test_results.csv 中错误样本的数量", "data"),
    ("对比 model_a 和 model_b 的 F1 score", "data"),
    
    # ✅ report: 汇总生成报告
    ("根据刚才的分析结果写一份总结报告", "report"),
    ("把这三篇论文的结论整理成表格", "report"),
    ("生成本周实验进展汇报", "report"),
    ("汇总所有模型的评估指标并给出推荐", "report"),
    ("将检索到的内容整理成文献综述", "report"),
    
    # ✅ general: 普通对话/兜底
    ("你好", "general"),
    ("今天天气怎么样", "general"),
    ("你是谁开发的", "general"),
    ("谢谢你的帮助", "general"),
    ("帮我写一首关于人工智能的诗", "general"),
]


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("question,expected_route", ROUTING_TEST_CASES)
async def test_router_classification(question: str, expected_route: str):
    """参数化测试：验证 20 个问题的路由分类是否符合预期"""
    result = await route_question(question)
    assert result == expected_route, (
        f"问题: '{question}'\n"
        f"期望: {expected_route}\n"
        f"实际: {result}"
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_router_fallback_to_general():
    """验证无法识别的问题应兜底返回 general"""
    # 使用一个模糊/无意义的问题测试白名单兜底逻辑
    result = await route_question("asdfjkl;🚀💯")
    assert result == "general", f"异常输入应兜底为 general，实际返回: {result}"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_router_returns_valid_literal():
    """验证返回值始终是合法的 Literal 类型"""
    valid_routes = {"retrieval", "data", "report", "general"}
    test_questions = ["查论文", "算MAE", "写报告", "你好"]
    for q in test_questions:
        result = await route_question(q)
        assert result in valid_routes, f"返回了非法路由标签: {result}"