from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.services.langchain_llm import get_langchain_llm


llm = get_langchain_llm()

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是简洁、严谨的科研助手。"),
    ("human", "请用一句话解释：{topic}"),
])

# LCEL：Prompt -> LLM -> 字符串输出
explain_chain = explain_prompt | llm | StrOutputParser()


class RouteResult(BaseModel):
    route: Literal["retrieval", "data", "general"] = Field(
        description="问题应该走的处理路径"
    )
    reason: str = Field(description="一句话说明分类原因")


route_parser = PydanticOutputParser(pydantic_object=RouteResult)
route_prompt = ChatPromptTemplate.from_messages([
    ("system", "你负责给问题分类。必须严格按指定 JSON 格式输出。"),
    (
        "human",
        "问题：{question}\n\n{format_instructions}",
    ),
]).partial(format_instructions=route_parser.get_format_instructions())

route_chain = route_prompt | llm | route_parser