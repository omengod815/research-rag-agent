from fastapi import APIRouter
from app.schemas.rag import RagRequest
from app.rag.pipeline import rag_answer

router = APIRouter(prefix="/rag", tags=["rag"])

@router.post("/query")
async def query_rag(req: RagRequest):
    return await rag_answer(req.question, req.top_k)