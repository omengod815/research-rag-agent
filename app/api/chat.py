from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import chat_once
from app.core.config import get_settings

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        answer = await chat_once(req.message)
        return ChatResponse(
            answer=answer,
            model=get_settings().qwen_chat_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc