from app.services.llm_service import chat_once

async def build_report(question: str, evidence: str, analysis: str = "") -> str:
    prompt = f"""
你是科研报告助手。请基于证据与分析结果生成简洁报告。
必须区分“资料直接支持”和“推断/建议”。不要编造不存在的数据。

用户问题：{question}

文献证据：
{evidence}

数据分析：
{analysis or '无'}

输出格式：
1. 结论
2. 关键证据
3. 数据结果（如有）
4. 局限与下一步
""".strip()
    return await chat_once(prompt)