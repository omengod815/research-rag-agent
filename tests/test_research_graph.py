"""
Research Graph 单元测试
覆盖 general / data 两条路由分支及异常兜底逻辑
"""

import pytest
from pathlib import Path

from app.agents.research_graph import research_graph
from app.tools.data_tool import SAFE_DIR


@pytest.mark.asyncio
async def test_general_route():
    """测试通用问答分支：无文件名时应走 general 路由"""
    state = await research_graph.ainvoke({
        "question": "什么是线性回归？",
        "filename": ""
    })

    assert state["route"] == "general"
    assert "answer" in state
    assert len(state["answer"]) > 0


@pytest.mark.asyncio
async def test_data_route_with_file():
    """测试数据分析分支：使用 SAFE_DIR 内的合法 CSV 验证指标计算"""
    csv_file = SAFE_DIR / "test_model.csv"
    # 使用 y_true/y_pred 列名，验证列名兼容逻辑
    csv_file.write_text(
        "y_true,y_pred\n1,1.1\n2,2.2\n3,2.8\n",
        encoding="utf-8"
    )

    try:
        state = await research_graph.ainvoke({
            "question": f"帮我分析 {csv_file.name} 的预测效果",
            "filename": csv_file.name  # 只传文件名，data_tool 内部拼接 SAFE_DIR
        })

        assert state["route"] == "data"
        analysis = state.get("analysis", "")
        
        # 不应包含失败提示
        assert "⚠️" not in analysis, f"分析意外失败: {analysis}"
        # 应包含至少一个回归指标
        assert any(m in analysis for m in ["RMSE", "MAE", "R2", "MSE"]), \
            f"分析结果缺少回归指标，实际内容: {analysis}"
        assert "answer" in state and len(state["answer"]) > 0
    finally:
        # 清理测试文件，避免污染生产数据目录
        if csv_file.exists():
            csv_file.unlink()


@pytest.mark.asyncio
async def test_data_route_missing_file():
    """测试数据分析分支兜底：文件不存在时 Graph 不应崩溃，应返回友好提示"""
    state = await research_graph.ainvoke({
        "question": "分析不存在的文件",
        "filename": "nonexistent_file_12345.csv"
    })

    assert state["route"] == "data"
    assert "answer" in state
    assert len(state["answer"]) > 0
    
    # Graph 不应泄露原始堆栈
    assert "Traceback" not in state["answer"]
    # analysis 字段应包含我们在 data_node 中定义的友好错误提示
    assert "⚠️" in state.get("analysis", "") or "not found" in state.get("analysis", "").lower()