import asyncio

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)

from app.core.config import get_settings


async def chat_once(message: str) -> str:
    settings = get_settings()

    client = AsyncOpenAI(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,

        # 单次 Qwen 请求最长等待 120 秒
        timeout=120.0,

        # 关闭 SDK 自带重试，下面自己控制
        max_retries=0,
    )

    max_attempts = 4

    try:
        for attempt in range(1, max_attempts + 1):
            try:
                response = await client.chat.completions.create(
                    model=settings.qwen_chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是严谨的科研论文问答助手。"
                                "请优先依据用户提供的检索资料回答，"
                                "资料不足时明确说明资料不足，"
                                "不要编造论文中不存在的信息。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": message,
                        },
                    ],
                    temperature=0.2,
                )

                content = response.choices[0].message.content

                return content or ""

            except (
                APITimeoutError,
                APIConnectionError,
                RateLimitError,
            ) as exc:

                if attempt >= max_attempts:
                    print(
                        f"LLM request failed after "
                        f"{max_attempts} attempts."
                    )
                    print(
                        f"Reason: {type(exc).__name__}"
                    )
                    raise

                wait_seconds = 2 ** attempt

                print(
                    f"LLM request failed. "
                    f"Retry {attempt}/{max_attempts} "
                    f"after {wait_seconds}s. "
                    f"Reason: {type(exc).__name__}"
                )

                await asyncio.sleep(wait_seconds)

        raise RuntimeError(
            "LLM request failed unexpectedly."
        )

    finally:
        await client.close()