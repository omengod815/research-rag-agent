import logging
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agents.report_agent import build_report
from app.agents.router import route_question
from app.rag.retriever import retrieve
from app.services.llm_service import chat_once
from app.tools.data_tool import analyze_regression_csv_impl

logger = logging.getLogger(__name__)
DB_PATH = Path("data/checkpoints.sqlite")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class ResearchState(TypedDict, total=False):
    question: str
    filename: str
    thread_id: str
    user_id: str
    route: str
    evidence: str
    sources: list[dict]
    analysis: str
    answer: str
    errors: list[str]
    approved: bool


def _with_error(state: ResearchState, message: str) -> list[str]:
    return [*state.get("errors", []), message]


async def router_node(state: ResearchState) -> dict:
    try:
        return {"route": await route_question(state["question"])}
    except Exception:
        logger.exception("router_node failed")
        return {"route": "general", "errors": _with_error(state, "Router 执行失败，已降级为普通回答。")}


async def retrieval_node(state: ResearchState) -> dict:
    try:
        hits = await retrieve(
            state["question"],
            top_k=5,
            user_id=state.get("user_id", "local-demo"),
        )
        parts = [
            f"[{i}] source={h.get('source', 'unknown')} page={h.get('page', 'unknown')}\n{h.get('text', '')}"
            for i, h in enumerate(hits, start=1)
        ]
        return {
            "sources": hits,
            "evidence": "\n\n".join(parts) if parts else "没有检索到相关证据。",
        }
    except Exception:
        logger.exception("retrieval_node failed")
        return {
            "sources": [],
            "evidence": "检索工具执行失败，当前没有可用的文献证据。",
            "errors": _with_error(state, "Retrieval Tool 执行失败。"),
        }


def data_node(state: ResearchState) -> dict:
    filename = state.get("filename", "").strip()
    if not filename:
        return {
            "analysis": "用户没有提供 CSV 文件名，因此未执行数值分析。",
            "errors": _with_error(state, "Data Tool 未执行：缺少 filename。"),
        }
    try:
        return {"analysis": analyze_regression_csv_impl(filename)}
    except Exception as exc:
        logger.exception("data_node failed")
        return {
            "analysis": f"CSV 分析失败：{type(exc).__name__}。请检查目录、文件名与列名。",
            "errors": _with_error(state, f"Data Tool 执行失败：{type(exc).__name__}。"),
        }


async def general_node(state: ResearchState) -> dict:
    return {"evidence": "本次未执行文献检索。", "analysis": "本次未执行 CSV 数据分析。", "sources": []}


def approval_node(state: ResearchState) -> dict:
    """interrupt 会保存检查点并暂停；恢复时传入 Command(resume=True/False)。"""
    decision = interrupt({
        "kind": "approval",
        "message": "即将读取并分析受控目录中的 CSV。是否批准继续？",
        "filename": state.get("filename", ""),
        "thread_id": state.get("thread_id", ""),
    })
    approved = bool(decision)
    if not approved:
        return {
            "approved": False,
            "analysis": "用户拒绝了本次 CSV 分析，Data Tool 未执行。",
        }
    return {"approved": True}


async def report_node(state: ResearchState) -> dict:
    if state.get("route") == "general":
        prompt = f"你是简洁、严谨的科研助手。请直接回答普通问题；不确定时明确说明。\n问题：{state['question']}"
        answer = await chat_once(prompt)
    else:
        answer = await build_report(
            question=state["question"],
            evidence=state.get("evidence", ""),
            analysis=state.get("analysis", ""),
            errors=state.get("errors", []),
        )
    return {"answer": answer}


def after_router(state: ResearchState) -> str:
    route = state.get("route", "general")
    if route in {"retrieval", "combined"}:
        return "retrieval"
    if route == "data":
        return "approval"
    return "general"


def after_retrieval(state: ResearchState) -> str:
    return "approval" if state.get("route") == "combined" else "report"


def after_approval(state: ResearchState) -> str:
    return "data" if state.get("approved") else "report"


def build_research_graph(checkpointer=None, require_approval: bool = False):
    builder = StateGraph(ResearchState)
    builder.add_node("router", router_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("data", data_node)
    builder.add_node("general", general_node)
    builder.add_node("report", report_node)

    if require_approval:
        builder.add_node("approval", approval_node)

    builder.add_edge(START, "router")
    if require_approval:
        builder.add_conditional_edges(
            "router", after_router,
            {"retrieval": "retrieval", "approval": "approval", "general": "general"},
        )
        builder.add_conditional_edges(
            "retrieval", after_retrieval,
            {"approval": "approval", "report": "report"},
        )
        builder.add_conditional_edges(
            "approval", after_approval,
            {"data": "data", "report": "report"},
        )
    else:
        # 阶段 19 的行为：不暂停，直接执行 Data Tool。
        def after_router_without_approval(state: ResearchState) -> str:
            route = state.get("route", "general")
            if route in {"retrieval", "combined"}:
                return "retrieval"
            return "data" if route == "data" else "general"

        def after_retrieval_without_approval(state: ResearchState) -> str:
            return "data" if state.get("route") == "combined" else "report"

        builder.add_conditional_edges(
            "router", after_router_without_approval,
            {"retrieval": "retrieval", "data": "data", "general": "general"},
        )
        builder.add_conditional_edges(
            "retrieval", after_retrieval_without_approval,
            {"data": "data", "report": "report"},
        )

    builder.add_edge("data", "report")
    builder.add_edge("general", "report")
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer)


# 保留无审批的图对象，便于阶段 19 单元测试和本地对比。
research_graph = build_research_graph()


async def invoke_with_checkpoint(inputs: dict, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_research_graph(checkpointer=checkpointer, require_approval=True)
        return await graph.ainvoke({**inputs, "thread_id": thread_id}, config=config)


async def resume_with_checkpoint(thread_id: str, approved: bool) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_research_graph(checkpointer=checkpointer, require_approval=True)
        return await graph.ainvoke(Command(resume=approved), config=config)


async def stream_with_checkpoint(inputs: dict, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        graph = build_research_graph(checkpointer=checkpointer, require_approval=True)
        async for update in graph.astream(
            {**inputs, "thread_id": thread_id},
            config=config,
            stream_mode="updates",
        ):
            yield update
