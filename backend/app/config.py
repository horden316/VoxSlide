from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///./storage/app.db"
    storage_dir: Path = Path("./storage")
    backend_cors_origins: str = "*"
    tts_provider: str = "qwen_local"
    openai_api_key: str | None = None
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    qwen_tts_endpoint: str | None = "http://localhost:7860/tts"
    qwen_tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    qwen_tts_voice: str = "Ryan"
    qwen_tts_voices: str = "Ryan:Ryan,Aiden:Aiden,Vivian:Vivian,Serena:Serena,Uncle_Fu:Uncle Fu,Dylan:Dylan,Eric:Eric,Ono_Anna:Ono Anna,Sohee:Sohee"
    video_width: int = Field(default=1920)
    video_height: int = Field(default=1080)
    video_encoder: str = "h264_nvenc"
    video_segment_workers: int = Field(default=4, ge=1)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
