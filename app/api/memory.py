from fastapi import APIRouter

from app.agents.memory_graph import chat_with_memory
from app.schemas.memory import MemoryChatRequest

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("/chat")
async def memory_chat(req: MemoryChatRequest):
    answer = await chat_with_memory(req.thread_id, req.message)
    return {"thread_id": req.thread_id, "answer": answer}