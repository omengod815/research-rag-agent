from app.services.llm_service import chat_once


async def build_report(
    question: str,
    evidence: str,
    analysis: str = "",
    errors: list[str] | None = None,
) -> str:
    """只根据已提供的证据和计算结果整理报告，不负责自行调用工具。"""
    error_text = "\n".join(f"- {item}" for item in (errors or [])) or "无"
    prompt = f"""
你是严谨的科研助理。请回答用户问题，并遵守以下约束：
1. 不得编造资料中没有的论文内容、页码、指标或数字。
2. 区分“检索证据直接支持的结论”和“基于证据作出的解释”。
3. 如果证据或数据不足，要明确说不足，不要猜测。
4. 若有文献证据，引用对应的 [1]、[2] 编号；不要声称某段证据支持它没有提到的结论。
5. 如果某个工具失败，说明哪些内容没有成功获取。

用户问题：
{question}

检索证据：
{evidence or '本次没有检索到文献证据。'}

数据分析结果：
{analysis or '本次没有数据分析结果。'}

工具错误/提醒：
{error_text}

请按以下结构组织回答：
1. 结论
2. 关键证据
3. 数据结果（如有）
4. 局限与下一步
""".strip()
    return await chat_once(prompt)
