from datetime import datetime
from typing import Any
from pydantic import BaseModel


class ProjectCreate(BaseModel):
    title: str


class ProjectOut(BaseModel):
    id: int
    title: str
    original_pdf_path: str | None
    output_video_path: str | None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PagePatch(BaseModel):
    transcript: str


class GenerateAudioRequest(BaseModel):
    provider: str | None = None
    voice: str | None = None
    language: str | None = None
    instruct: str | None = None
    tts_params: dict[str, Any] | None = None


class RenderVideoRequest(BaseModel):
    provider: str | None = None
    voice: str | None = None
    language: str | None = None
    instruct: str | None = None
    tts_params: dict[str, Any] | None = None
    force_regenerate: bool = False


class PageOut(BaseModel):
    id: int
    project_id: int
    page_number: int
    image_url: str
    transcript: str
    audio_url: str | None
    audio_duration: float | None
    created_at: datetime
    updated_at: datetime


class JobOut(BaseModel):
    id: int
    project_id: int
    type: str
    status: str
    progress: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JobCreated(BaseModel):
    job_id: int


class TtsVoiceOut(BaseModel):
    id: str
    label: str


class TtsConfigOut(BaseModel):
    provider: str
    model: str
    default_voice: str
    voices: list[TtsVoiceOut]
    params: dict[str, Any] | None = None
    speaker_instructs: dict[str, str] | None = None
