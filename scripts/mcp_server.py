import sys
from pathlib import Path

# 把项目根目录加到 Python 路径，这样在 scripts/ 下运行也能找到 app 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server import MCPServer

from app.rag.retriever import retrieve
from app.tools.data_tool import analyze_regression_csv_impl

mcp = MCPServer("Research RAG Tools")


@mcp.tool()
async def search_research_papers(query: str) -> str:
    """Search the local research knowledge base and return evidence with source and page."""
    hits = await retrieve(query, top_k=5, user_id="local-demo", mode="hybrid")
    if not hits:
        return "没有检索到证据。请检查是否已上传论文并完成索引。"
    return "\n\n".join(
        f"[{index}] source={hit.get('source')} page={hit.get('page')} "
        f"score={hit.get('rerank_score', hit.get('score', 0)):.4f}\n{hit.get('text', '')}"
        for index, hit in enumerate(hits, start=1)
    )


@mcp.tool()
def analyze_csv(filename: str) -> str:
    """Calculate MAE, RMSE, and R2 for a CSV in the controlled data/public directory."""
    return analyze_regression_csv_impl(filename)


if __name__ == "__main__":
    # Streamable HTTP 默认 MCP 端点是 /mcp。
    mcp.run(transport="streamable-http", host="127.0.0.1", port=3001)