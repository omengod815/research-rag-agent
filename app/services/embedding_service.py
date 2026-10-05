import asyncio

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)

from app.core.config import get_settings


async def embed_texts(
    texts: list[str],
) -> list[list[float]]:

    settings = get_settings()

    client = AsyncOpenAI(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,

        # 单次请求最多等 90 秒
        timeout=90.0,

        # 我们自己做重试，关闭 SDK 内部重试
        max_retries=0,
    )

    max_attempts = 4

    try:
        for attempt in range(
            1,
            max_attempts + 1,
        ):
            try:
                response = await client.embeddings.create(
                    model=settings.qwen_embedding_model,
                    input=texts,
                    dimensions=(
                        settings.qwen_embedding_dimensions
                    ),
                )

                return [
                    item.embedding
                    for item in response.data
                ]

            except (
                APITimeoutError,
                APIConnectionError,
                RateLimitError,
            ) as exc:

                if attempt >= max_attempts:
                    print(
                        "Embedding failed after "
                        f"{max_attempts} attempts: {exc}"
                    )
                    raise

                # 2、4、8 秒逐渐增加
                wait_seconds = 2 ** attempt

                print(
                    "Embedding request failed. "
                    f"Retry {attempt}/{max_attempts} "
                    f"after {wait_seconds}s. "
                    f"Reason: {type(exc).__name__}"
                )

                await asyncio.sleep(
                    wait_seconds
                )

        raise RuntimeError(
            "Embedding failed unexpectedly."
        )

    finally:
        await client.close()