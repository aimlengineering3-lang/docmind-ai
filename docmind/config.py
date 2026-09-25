from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-lite-latest"
    pinecone_api_key: str = ""
    pinecone_index: str = "docmind"


@lru_cache
def get_settings() -> Settings:
    return Settings()