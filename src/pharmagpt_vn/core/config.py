from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_port: int = 8003
    log_level: str = "INFO"

    openai_base_url: str
    openai_api_key: str
    llm_model_main: str

    embedding_model: str
    embedder_device: str = "cpu"

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    reranker_model: str = "cohere/rerank-v3.5"

    qu_rewrite_n: int = 3
    qu_decompose_max: int = 4
    qu_hyde_enabled: bool = True
    qu_hyde_max_tokens: int = 256
    qu_per_branch_top_k: int = 30
    qu_rrf_k: int = 60

    qdrant_url: str
    qdrant_api_key: str | None = None
    qdrant_collection: str

    redis_url: str
    internal_api_token: str

    enforce_citations_for_clinical: bool = True
    pii_redaction_enabled: bool = True
    medical_disclaimer_enabled: bool = True

    default_temperature: float = 0.2
    default_top_p: float = 0.9
    default_max_tokens: int = 512


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
