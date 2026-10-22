import hashlib
import logging

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.services.embedding_cache import get_cached_embedding, set_cached_embedding

logger = logging.getLogger(__name__)


def _cache_key(text: str, model: str, dimensions: int) -> str:
    raw = f"{model}|{dimensions}|{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    settings = get_settings()
    output: list[list[float] | None] = [None] * len(texts)
    missing: dict[str, tuple[str, list[int]]] = {}

    # 先查缓存；同一批中重复出现的文本也只放入待请求列表一次。
    for index, text in enumerate(texts):
        key = _cache_key(text, settings.qwen_embedding_model, settings.qwen_embedding_dimensions)
        cached = get_cached_embedding(key)
        if cached is not None:
            output[index] = cached
        elif key in missing:
            original_text, indices = missing[key]
            indices.append(index)
            missing[key] = (original_text, indices)
        else:
            missing[key] = (text, [index])

    if missing:
        client = AsyncOpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url)
        keys = list(missing.keys())
        unique_texts = [missing[key][0] for key in keys]
        response = await client.embeddings.create(
            model=settings.qwen_embedding_model,
            input=unique_texts,
            dimensions=settings.qwen_embedding_dimensions,
        )
        ordered_items = sorted(response.data, key=lambda item: getattr(item, "index", 0))
        if len(ordered_items) != len(keys):
            raise RuntimeError("Embedding API 返回数量与输入文本数量不一致。")

        for key, item in zip(keys, ordered_items):
            vector = list(item.embedding)
            if len(vector) != settings.qwen_embedding_dimensions:
                raise RuntimeError(
                    f"Embedding 维度不匹配：期望 {settings.qwen_embedding_dimensions}，实际 {len(vector)}。"
                )
            set_cached_embedding(key, vector)
            for index in missing[key][1]:
                output[index] = list(vector)

    if any(vector is None for vector in output):
        raise RuntimeError("Embedding 缓存流程未能填满全部输出。")
    return [vector for vector in output if vector is not None]