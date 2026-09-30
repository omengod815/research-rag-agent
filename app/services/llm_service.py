from openai import AsyncOpenAI
from app.core.config import get_settings


async def chat_once(message: str) -> str:
    settings = get_settings()
    client = AsyncOpenAI(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
    )

    response = await client.chat.completions.create(
        model=settings.qwen_chat_model,
        messages=[
            {"role": "system", "content": "你是简洁、严谨的科研助手。"},
            {"role": "user", "content": message},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""