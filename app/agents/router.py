from typing import Literal
from pydantic import BaseModel
from app.services.llm_service import chat_once

class RouteResult(BaseModel):
    route: Literal["retrieval", "data", "report", "general"]

async def route_question(question: str) -> str:
    prompt = f"""
把用户问题严格分类成以下一个标签：
retrieval：需要查询论文/知识库
 data：需要分析 CSV/数值
 report：需要汇总已有结果生成报告
 general：普通对话
只返回标签，不要解释。

问题：{question}
""".strip()
    raw = (await chat_once(prompt)).strip().lower()
    allowed = {"retrieval", "data", "report", "general"}
    return raw if raw in allowed else "general"