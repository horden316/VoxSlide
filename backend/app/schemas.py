from datetime import datetime
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
    voice: str | None = None


class RenderVideoRequest(BaseModel):
    voice: str | None = None


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
