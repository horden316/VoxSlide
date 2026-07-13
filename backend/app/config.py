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
    openai_tts_voices: str = "alloy:Alloy,ash:Ash,ballad:Ballad,coral:Coral,echo:Echo,fable:Fable,nova:Nova,onyx:Onyx,sage:Sage,shimmer:Shimmer,verse:Verse"
    qwen_tts_endpoint: str | None = "http://localhost:7860/tts"
    qwen_tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    qwen_tts_voice: str = "Ryan"
    qwen_tts_voices: str = "Ryan:Ryan,Aiden:Aiden,Vivian:Vivian,Serena:Serena,Uncle_Fu:Uncle Fu,Dylan:Dylan,Eric:Eric,Ono_Anna:Ono Anna,Sohee:Sohee"
    kokoro_tts_endpoint: str | None = "http://localhost:7861/tts"
    kokoro_tts_model: str = "hexgrad/Kokoro-82M"
    kokoro_tts_voice: str = "af_heart"
    bark_tts_endpoint: str | None = "http://localhost:7862/tts"
    bark_tts_model: str = "suno/bark"
    bark_tts_voice: str = "v2/en_speaker_6"
    bark_tts_voices: str = (
        "v2/en_speaker_6:English Speaker 6,v2/en_speaker_9:English Speaker 9,v2/en_speaker_3:English Speaker 3,"
        "v2/zh_speaker_1:Chinese Speaker 1,v2/zh_speaker_4:Chinese Speaker 4,v2/zh_speaker_9:Chinese Speaker 9,"
        "v2/ja_speaker_3:Japanese Speaker 3,v2/ko_speaker_0:Korean Speaker 0"
    )
    kokoro_tts_voices: str = "af_heart:Heart (US female),af_bella:Bella (US female),af_nicole:Nicole (US female),af_sky:Sky (US female),am_michael:Michael (US male),am_fenrir:Fenrir (US male),am_puck:Puck (US male),bf_emma:Emma (UK female),bm_george:George (UK male),bm_fable:Fable (UK male)"
    video_width: int = Field(default=1920)
    video_height: int = Field(default=1080)
    video_encoder: str = "h264_nvenc"
    # Slide segments are still images; 6fps keeps players happy while cutting
    # encode time ~3x versus the ffmpeg default 25fps.
    video_fps: int = Field(default=6, ge=1)
    video_segment_workers: int = Field(default=4, ge=1)
    # Slightly above QWEN_TTS_WORKERS so uvicorn's random request routing
    # keeps every TTS worker busy; extra requests just queue there.
    tts_workers: int = Field(default=3, ge=1)
    # Kokoro service worker count (same env the service reads); render jobs
    # apply the same +1 oversubscription as the Qwen pairing above.
    kokoro_tts_workers: int = Field(default=4, ge=1)
    # Bark is slow and VRAM-hungry, so render jobs match its worker count
    # exactly: extra queued requests would risk the 300s synthesis timeout.
    bark_tts_workers: int = Field(default=1, ge=1)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
