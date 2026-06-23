from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import SessionLocal, get_db
from ..models import Job, Page
from ..config import get_settings
from ..schemas import GenerateAudioRequest, JobCreated, PageOut, PagePatch, TtsConfigOut
from ..services.tts_service import TtsService
from ..services.video_service import VideoService
from ..storage import project_dir, public_file_url


router = APIRouter(tags=["pages"])


def clean_tts_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def serialize_page(page: Page) -> PageOut:
    return PageOut(
        id=page.id,
        project_id=page.project_id,
        page_number=page.page_number,
        image_url=public_file_url(page.image_path) or "",
        transcript=page.transcript,
        audio_url=public_file_url(page.audio_path),
        audio_duration=page.audio_duration,
        created_at=page.created_at,
        updated_at=page.updated_at,
    )


@router.get("/api/projects/{project_id}/pages", response_model=list[PageOut])
def list_pages(project_id: int, db: Session = Depends(get_db)) -> list[PageOut]:
    pages = db.query(Page).filter(Page.project_id == project_id).order_by(Page.page_number).all()
    return [serialize_page(page) for page in pages]


@router.patch("/api/pages/{page_id}", response_model=PageOut)
def update_page(page_id: int, payload: PagePatch, db: Session = Depends(get_db)) -> PageOut:
    page = db.get(Page, page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    if page.transcript != payload.transcript:
        page.audio_path = None
        page.audio_duration = None
    page.transcript = payload.transcript
    db.commit()
    db.refresh(page)
    return serialize_page(page)


@router.get("/api/tts/config", response_model=TtsConfigOut)
def get_tts_config() -> TtsConfigOut:
    settings = get_settings()
    service = TtsService()
    return TtsConfigOut(
        provider=settings.tts_provider,
        model=service.model,
        default_voice=service.default_voice,
        voices=service.available_voices(),
    )


@router.post("/api/pages/{page_id}/generate-audio", response_model=PageOut)
def generate_audio(page_id: int, payload: GenerateAudioRequest | None = None, db: Session = Depends(get_db)) -> PageOut:
    page = db.get(Page, page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    if not page.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required before generating audio")
    audio_path = project_dir(page.project_id) / "audio" / f"page-{page.page_number:04d}.mp3"
    try:
        TtsService().synthesize(
            page.transcript,
            audio_path,
            clean_tts_value(payload.voice) if payload else None,
            clean_tts_value(payload.language) if payload else None,
            clean_tts_value(payload.instruct) if payload else None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    page.audio_path = str(audio_path)
    page.audio_duration = VideoService().probe_duration(audio_path)
    db.commit()
    db.refresh(page)
    return serialize_page(page)


@router.post("/api/pages/{page_id}/generate-audio-job", response_model=JobCreated)
def generate_audio_job(
    page_id: int,
    background_tasks: BackgroundTasks,
    payload: GenerateAudioRequest | None = None,
    db: Session = Depends(get_db),
) -> JobCreated:
    page = db.get(Page, page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    if not page.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required before generating audio")
    job = Job(project_id=page.project_id, type=f"generate_audio:{page.id}", status="queued", progress=0)
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(
        run_generate_audio_job,
        job.id,
        page.id,
        clean_tts_value(payload.voice) if payload else None,
        clean_tts_value(payload.language) if payload else None,
        clean_tts_value(payload.instruct) if payload else None,
    )
    return JobCreated(job_id=job.id)


def update_audio_job(db: Session, job: Job, status: str | None = None, progress: int | None = None, error: str | None = None) -> None:
    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = max(0, min(100, progress))
    if error is not None:
        job.error_message = error
    db.commit()
    db.refresh(job)


def run_generate_audio_job(
    job_id: int,
    page_id: int,
    voice: str | None = None,
    language: str | None = None,
    instruct: str | None = None,
) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        page = db.get(Page, page_id)
        if not job or not page:
            return
        update_audio_job(db, job, status="running", progress=5)
        if not page.transcript.strip():
            raise RuntimeError("Transcript is required before generating audio")

        audio_path = project_dir(page.project_id) / "audio" / f"page-{page.page_number:04d}.mp3"
        update_audio_job(db, job, progress=20)
        TtsService().synthesize(page.transcript, audio_path, voice, language, instruct)
        update_audio_job(db, job, progress=85)
        page.audio_path = str(audio_path)
        page.audio_duration = VideoService().probe_duration(audio_path)
        db.commit()
        update_audio_job(db, job, status="completed", progress=100)
    except Exception as exc:
        job = db.get(Job, job_id)
        if job:
            update_audio_job(db, job, status="failed", error=str(exc))
    finally:
        db.close()
