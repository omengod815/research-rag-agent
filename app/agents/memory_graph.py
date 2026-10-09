from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.services.langchain_llm import get_langchain_llm

DB_PATH = Path("data/checkpoints.sqlite")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class MemoryState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async def model_node(state: MemoryState):
    response = await get_langchain_llm().ainvoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MemoryState)
builder.add_node("model", model_node)
builder.add_edge(START, "model")
builder.add_edge("model", END)


async def chat_with_memory(thread_id: str, message: str) -> str:
    # SQLite 文件会保留不同 thread_id 的 checkpoint。
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        return result["messages"][-1].content