import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from app.agents.research_graph import (
    invoke_with_checkpoint,
    resume_with_checkpoint,
    stream_with_checkpoint,
)
from app.schemas.agent import AgentRequest, AgentResumeRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


def _public_result(result: dict, thread_id: str) -> dict:
    interrupts = result.get("__interrupt__") or []
    if interrupts:
        values = [getattr(item, "value", str(item)) for item in interrupts]
        return {"status": "approval_required", "thread_id": thread_id, "interrupts": values}
    return {
        "status": "completed",
        "thread_id": thread_id,
        "route": result.get("route", "unknown"),
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "evidence": result.get("evidence", ""),
        "analysis": result.get("analysis", ""),
        "errors": result.get("errors", []),
    }


@router.post("/run")
async def run_agent(req: AgentRequest):
    try:
        result = await invoke_with_checkpoint(
            {"question": req.question, "filename": req.filename},
            thread_id=req.thread_id,
        )
        return _public_result(result, req.thread_id)
    except Exception as exc:
        logger.exception("/agent/run failed")
        raise HTTPException(status_code=500, detail="Agent 执行失败，请检查服务日志。") from exc


@router.post("/resume")
async def resume_agent(req: AgentResumeRequest):
    try:
        result = await resume_with_checkpoint(req.thread_id, req.approved)
        return _public_result(result, req.thread_id)
    except Exception as exc:
        logger.exception("/agent/resume failed")
        raise HTTPException(status_code=500, detail="恢复工作流失败；请确认 thread_id 曾经暂停。") from exc


@router.post("/stream")
async def stream_agent(req: AgentRequest):
    async def event_generator():
        try:
            async for update in stream_with_checkpoint(
                {"question": req.question, "filename": req.filename},
                thread_id=req.thread_id,
            ):
                data = json.dumps(jsonable_encoder(update), ensure_ascii=False)
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception:
            logger.exception("/agent/stream failed")
            data = json.dumps({"error": "Streaming 执行失败，请检查服务日志。"}, ensure_ascii=False)
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
