from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_port: int = 8003
    log_level: str = "INFO"

    base_model: str
    adapter_path: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.85

    embedding_model: str

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
