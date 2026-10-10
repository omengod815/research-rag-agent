from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.rag import router as rag_router
from app.api.memory import router as memory_router
from app.api.agent import router as agent_router

app = FastAPI(title="Research RAG Agent", version="0.3.0")

app.include_router(agent_router)
app.include_router(memory_router)
app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(documents_router)

@app.get("/health")
async def health():
    return {"status": "ok"}