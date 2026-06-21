from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///./storage/app.db"
    storage_dir: Path = Path("./storage")
    backend_cors_origins: str = "http://localhost:3000"
    openai_api_key: str | None = None
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    video_width: int = Field(default=1920)
    video_height: int = Field(default=1080)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
