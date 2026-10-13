from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

# === 保留你原有的业务导入 ===
from app.agents.router import route_question
from app.rag.retriever import retrieve
from app.agents.report_agent import build_report
from app.tools.data_tool import analyze_regression_csv_impl


class ResearchState(TypedDict, total=False):
    question: str
    filename: str
    route: str
    evidence: str
    analysis: str
    answer: str


# === 原有节点保持不变 ===
async def router_node(state: ResearchState):
    return {"route": await route_question(state["question"])}


async def retrieval_node(state: ResearchState):
    hits = await retrieve(state["question"], top_k=5)
    evidence = "\n\n".join(
        f"{h['source']} p.{h['page']}\n{h['text']}" for h in hits
    )
    return {"evidence": evidence}


async def general_node(state: ResearchState):
    return {"evidence": "无外部检索证据", "analysis": ""}


async def report_node(state: ResearchState):
    answer = await build_report(
        state["question"],
        state.get("evidence", ""),
        state.get("analysis", ""),
    )
    return {"answer": answer}


def after_router(state: ResearchState):
    route = state.get("route", "general")
    if route == "retrieval":
        return "retrieval"
    if route == "data":
        return "data"
    return "general"


# === 🔴 核心修改：data_node 加入 HITL 审批 ===
async def data_node(state: ResearchState):
    filename = state.get("filename", "")
    if not filename:
        return {"analysis": "用户未提供 CSV 文件名。"}

    # 暂停 Graph，向前端抛出审批请求
    decision = interrupt({
        "type": "approval",
        "message": f"⚠️ 即将执行高风险操作：分析文件 [{filename}]。是否继续？",
        "filename": filename
    })

    # 接收前端传回的审批结果
    if not decision or not decision.get("approved"):
        return {"analysis": "❌ 用户拒绝了数据分析操作。"}

    try:
        result = analyze_regression_csv_impl(filename)
        return {"analysis": result}
    except Exception as e:
        return {"analysis": f"⚠️ 数据分析失败: {str(e)}"}


# === 构建 Graph ===
builder = StateGraph(ResearchState)
builder.add_node("router", router_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("data", data_node)
builder.add_node("general", general_node)
builder.add_node("report", report_node)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    after_router,
    {
        "retrieval": "retrieval",
        "data": "data",
        "general": "general",
    },
)
builder.add_edge("retrieval", "report")
builder.add_edge("data", "report")
builder.add_edge("general", "report")
builder.add_edge("report", END)

# === 🔴 核心修改：编译时注入 MemorySaver ===
memory = MemorySaver()
research_graph = builder.compile(checkpointer=memory)