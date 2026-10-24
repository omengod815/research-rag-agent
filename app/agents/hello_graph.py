from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class DemoState(TypedDict):
    message: str
    result: str

def process(state: DemoState):
    return {"result": f"processed: {state['message']}"}

builder = StateGraph(DemoState)
builder.add_node("process", process)
builder.add_edge(START, "process")
builder.add_edge("process", END)

graph = builder.compile()