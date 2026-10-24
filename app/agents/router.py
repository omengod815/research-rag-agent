from app.services.llm_service import chat_once

ALLOWED_ROUTES = {"retrieval", "data", "combined", "general"}


async def route_question(question: str) -> str:
    """将用户问题路由到一个白名单路径；模型不输出有效标签时走 general。"""
    prompt = f"""
你是 Research RAG Agent 的路由器。只能从以下四个标签中选择一个，且只输出标签本身：
- retrieval：问题需要查询已建立索引的论文/知识库。
- data：问题需要读取 CSV 并计算指标；用户提供了文件名或明确要求数值分析。
- combined：问题明确要求同时使用论文检索证据和 CSV 数据分析。
- general：普通解释、闲聊，或不需要查询论文/CSV 的问题。

不要回答用户的问题，不要输出 JSON，不要添加标点或解释。
用户问题：{question}
标签：
""".strip()

    raw = (await chat_once(prompt)).strip().lower()
    # 兼容模型意外输出反引号或额外说明的情况，但只接受白名单标签。
    first_line = raw.splitlines()[0].strip().strip("`'\".：: ") if raw else ""
    return first_line if first_line in ALLOWED_ROUTES else "general"
