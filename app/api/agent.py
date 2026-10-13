import json
import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from langgraph.types import Command

# === 保留你原有的导入 ===
from app.schemas.agent import AgentRequest
from app.agents.research_graph import research_graph

router = APIRouter(prefix="/agent", tags=["agent"])


# ================= 🔴 核心修复：全局拦截不可序列化对象 =================
class SafeJSONEncoder(json.JSONEncoder):
    """全局自定义编码器，遇到任何无法序列化的对象自动转为字符串，彻底杜绝 Interrupt 报错"""
    def default(self, obj):
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, (list, str, int, float, bool, type(None))):
            return obj
        # 尝试序列化 Pydantic 模型
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        # 最终兜底：转为字符串描述
        return f"<{type(obj).__name__} object: {str(obj)}>"

# 保存原始 dumps 方法，确保局部调用时也能用到自定义编码器
_original_json_dumps = json.dumps

def safe_dumps(*args, **kwargs):
    """强制使用自定义编码器，防止任何遗漏"""
    if 'cls' not in kwargs:
        kwargs['cls'] = SafeJSONEncoder
    return _original_json_dumps(*args, **kwargs)


# ================= 原有接口（完整保留） =================
@router.post("/run")
async def run_agent(req: AgentRequest):
    """注意：由于 graph 现在需要 checkpointer，
    如果原有 /run 接口也需要持久化状态，建议后续也加上 config。
    当前为保持向后兼容，使用临时 thread_id"""
    temp_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = await research_graph.ainvoke(
        {"question": req.question, "filename": req.filename},
        config=temp_config
    )
    return result


# ================= HITL + Streaming 接口 =================
class AgentStreamRequest(BaseModel):
    """扩展原有请求模型，增加 HITL 所需字段"""
    question: str
    filename: str = ""
    thread_id: str = ""
    resume_decision: Optional[dict] = None


@router.post("/stream")
async def agent_stream(req: AgentStreamRequest):
    """专用于 Gradio HITL 交互的流式接口"""
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 判断是首次运行还是中断后恢复
    if req.resume_decision is not None:
        input_data = Command(resume=req.resume_decision)
    else:
        input_data = {
            "question": req.question,
            "filename": req.filename,
        }

    async def event_generator():
        try:
            async for update in research_graph.astream(
                input_data, config=config, stream_mode="updates"
            ):
                # 使用 safe_dumps 替代 json.dumps，防止 update 里的嵌套对象触发崩溃
                yield f"data: {safe_dumps({'type': 'update', 'data': update}, ensure_ascii=False)}\n\n"

            # 检查 Graph 是否被 interrupt 暂停
            snapshot = await research_graph.aget_state(config)
            if snapshot.next:
                # 安全提取中断值（带空列表防御）
                raw_val = None
                if snapshot.tasks and snapshot.tasks[0].interrupts:
                    raw_val = snapshot.tasks[0].interrupts[0].value

                # safe_dumps 已经能完美处理 raw_val，直接传入即可
                yield f"data: {safe_dumps({'type': 'interrupt', 'data': raw_val, 'thread_id': thread_id}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {safe_dumps({'type': 'end', 'thread_id': thread_id}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {safe_dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")