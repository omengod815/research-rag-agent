from fastapi import FastAPI

# 创建一个 FastAPI 应用对象。以后 Uvicorn 启动的就是它。
app = FastAPI(
    title="Research RAG Agent",
    version="0.1.0"
)

# @app.get 表示：客户端 GET /health 时，执行下面的函数。
@app.get("/health")
async def health():
    # 返回 Python dict，FastAPI 会自动转换成 JSON。
    return {"status": "ok"}
