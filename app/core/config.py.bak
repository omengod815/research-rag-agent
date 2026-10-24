from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qwen_api_key: str
    qwen_base_url: str = Field(
        validation_alias=AliasChoices("QWEN_BASE_URL", "LLM_BASE_URL")
    )
    qwen_chat_model: str = Field(
        default="qwen3.7-flash",
        validation_alias=AliasChoices("QWEN_CHAT_MODEL", "LLM_MODEL"),
    )
    qwen_embedding_model: str = Field(
        default="qwen3.7-text-embedding-flash",
        validation_alias=AliasChoices("QWEN_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
    )
    qwen_embedding_dimensions: int = Field(
        default=1024,
        validation_alias=AliasChoices("QWEN_EMBEDDING_DIMENSIONS", "EMBEDDING_DIMENSIONS"),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
