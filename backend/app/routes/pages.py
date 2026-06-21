from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import Page
from ..schemas import PageOut, PagePatch
from ..services.tts_service import TtsService
from ..services.video_service import VideoService
from ..storage import project_dir, public_file_url


router = APIRouter(tags=["pages"])


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


@router.post("/api/pages/{page_id}/generate-audio", response_model=PageOut)
def generate_audio(page_id: int, db: Session = Depends(get_db)) -> PageOut:
    page = db.get(Page, page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    if not page.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required before generating audio")
    audio_path = project_dir(page.project_id) / "audio" / f"page-{page.page_number:04d}.mp3"
    try:
        TtsService().synthesize(page.transcript, audio_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    page.audio_path = str(audio_path)
    page.audio_duration = VideoService().probe_duration(audio_path)
    db.commit()
    db.refresh(page)
    return serialize_page(page)
