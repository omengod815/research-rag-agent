from fastapi import FastAPI
from app.api.chat import router as chat_router
app = FastAPI(title="Research RAG Agent", version="0.2.0")
app.include_router(chat_router)
@app.get("/health")
async def health():
    return {"status": "ok"}