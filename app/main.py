from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router

app = FastAPI(title="Research RAG Agent", version="0.2.0")
app.include_router(chat_router)
app.include_router(documents_router)
@app.get("/health")
async def health():
    return {"status": "ok"}