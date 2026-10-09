from langchain_openai import ChatOpenAI

from app.core.config import get_settings


def get_langchain_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
        model=settings.qwen_chat_model,
        temperature=0.2,
    )