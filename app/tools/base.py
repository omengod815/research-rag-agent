import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""


async def invoke_tool_safely(
    tool: BaseTool,
    payload: dict,
    timeout_seconds: float = 30.0,
) -> ToolResult:
    try:
        async with asyncio.timeout(timeout_seconds):
            value = await tool.ainvoke(payload)
        return ToolResult(ok=True, data=value)
    except TimeoutError:
        return ToolResult(ok=False, error=f"tool timeout: {tool.name}")
    except Exception as exc:
        # 对用户返回简洁错误；详细堆栈交给日志/Tracing。
        return ToolResult(ok=False, error=f"tool failed: {tool.name}: {exc}")