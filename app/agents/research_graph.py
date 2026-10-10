from typing import TypedDict
from langgraph.graph import StateGraph, START, END
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

async def router_node(state: ResearchState):
    return {"route": await route_question(state["question"])}

async def retrieval_node(state: ResearchState):
    hits = await retrieve(state["question"], top_k=5)
    evidence = "\n\n".join(
        f"{h['source']} p.{h['page']}\n{h['text']}" for h in hits
    )
    return {"evidence": evidence}

def data_node(state: ResearchState):
    filename = state.get("filename", "")
    if not filename:
        return {"analysis": "用户未提供 CSV 文件名。"}
    
    # 增加异常捕获，防止工具层报错导致整个 Graph 崩溃
    try:
        return {"analysis": analyze_regression_csv_impl(filename)}
    except (FileNotFoundError, ValueError, Exception) as e:
        return {"analysis": f"⚠️ 数据分析失败: {str(e)}"}

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

research_graph = builder.compile()