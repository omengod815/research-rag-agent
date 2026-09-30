from openai import AsyncOpenAI
from app.core.config import get_settings


async def embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    client = AsyncOpenAI(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
    )

    response = await client.embeddings.create(
        model=settings.qwen_embedding_model,
        input=texts,
        dimensions=settings.qwen_embedding_dimensions,
    )
    return [item.embedding for item in response.data]