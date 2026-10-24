from app.tools.retrieval_tool import search_papers

TOOLS = [search_papers]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

def get_tool(name: str):
    if name not in TOOLS_BY_NAME:
        raise KeyError(f"unknown tool: {name}")
    return TOOLS_BY_NAME[name]