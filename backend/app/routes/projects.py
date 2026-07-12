from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
from threading import Lock
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from ..config import get_settings
from ..database import SessionLocal, get_db
from ..models import Job, Page, Project
from ..schemas import JobCreated, ProjectCreate, ProjectOut, RenderVideoRequest
from ..services.pdf_service import PdfService
from ..services.tts_service import TtsService, timeline_sidecar_path
from ..services.video_service import VideoService
from ..storage import project_dir, safe_storage_path, unique_filename


router = APIRouter(prefix="/api/projects", tags=["projects"])


def clean_tts_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


@router.post("", response_model=ProjectOut)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> Project:
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Project title is required")
    project = Project(title=title)
    db.add(project)
    db.commit()
    db.refresh(project)
    project_dir(project.id)
    return project


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)) -> list[Project]:
    return db.query(Project).order_by(Project.created_at.desc()).all()


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: int, db: Session = Depends(get_db)) -> Project:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post("/{project_id}/upload-pdf", response_model=list[dict])
def upload_pdf(project_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)) -> list[dict]:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    filename = file.filename or ""
    if file.content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are accepted")
    header = file.file.read(5)
    file.file.seek(0)
    if header != b"%PDF-":
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF")

    base_dir = project_dir(project.id)
    pdf_path = base_dir / unique_filename(".pdf")
    with pdf_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    db.query(Page).filter(Page.project_id == project.id).delete()
    pages_dir = base_dir / "pages"
    if pages_dir.exists():
        shutil.rmtree(pages_dir)
    rendered = PdfService(get_settings().video_width, get_settings().video_height).render_pages(pdf_path, pages_dir)
    for index, image_path in enumerate(rendered, start=1):
        db.add(Page(project_id=project.id, page_number=index, image_path=str(image_path)))
    project.original_pdf_path = str(pdf_path)
    project.status = "pdf_uploaded"
    db.commit()
    return [{"page_number": i + 1, "image_url": f"/api/files/projects/{project.id}/pages/page-{i + 1:04d}.png"} for i in range(len(rendered))]


@router.post("/{project_id}/render-video", response_model=JobCreated)
def render_video(
    project_id: int,
    background_tasks: BackgroundTasks,
    payload: RenderVideoRequest | None = None,
    db: Session = Depends(get_db),
) -> JobCreated:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    pages = db.query(Page).filter(Page.project_id == project_id).order_by(Page.page_number).all()
    if not pages:
        raise HTTPException(status_code=400, detail="Upload a PDF before rendering")
    missing = [page.page_number for page in pages if not page.transcript.strip()]
    if missing:
        raise HTTPException(status_code=400, detail=f"Transcript is required for pages: {missing}")
    job = Job(project_id=project_id, type="render_video", status="queued", progress=0)
    project.status = "render_queued"
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(
        run_render_video_job,
        job.id,
        clean_tts_value(payload.voice) if payload else None,
        clean_tts_value(payload.language) if payload else None,
        clean_tts_value(payload.instruct) if payload else None,
        payload.force_regenerate if payload else False,
    )
    return JobCreated(job_id=job.id)


@router.get("/{project_id}/download")
def download_video(project_id: int, db: Session = Depends(get_db)) -> FileResponse:
    project = db.get(Project, project_id)
    if not project or not project.output_video_path:
        raise HTTPException(status_code=404, detail="Video not found")
    path = safe_storage_path(project.output_video_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path, media_type="video/mp4", filename=f"project-{project.id}-final.mp4")


@router.get("/{project_id}/download-srt")
def download_subtitle(project_id: int, db: Session = Depends(get_db)) -> FileResponse:
    project = db.get(Project, project_id)
    if not project or not project.output_video_path:
        raise HTTPException(status_code=404, detail="Subtitle not found")
    path = safe_storage_path(Path(project.output_video_path).with_suffix(".srt"))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Subtitle file not found")
    return FileResponse(path, media_type="application/x-subrip", filename=f"project-{project.id}-final.srt")


def load_timeline(audio_path: Path) -> list[dict] | None:
    """Read the per-chunk timing sidecar saved next to synthesized audio, if any."""
    sidecar = timeline_sidecar_path(audio_path)
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, list) and data else None


def update_job(db: Session, job: Job, status: str | None = None, progress: int | None = None, error: str | None = None) -> None:
    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = max(0, min(100, progress))
    if error is not None:
        job.error_message = error
    db.commit()
    db.refresh(job)


class TtsProgressAggregator:
    """Combines per-chunk progress from concurrently synthesized pages into one job progress value."""

    def __init__(self, job_id: int, total_steps: int):
        self.job_id = job_id
        self.total_steps = total_steps
        self.lock = Lock()
        self.fractions: dict[int, float] = {}

    def reporter(self, page_number: int):
        def report(completed_chunks: int, total_chunks: int) -> None:
            self.set_fraction(page_number, completed_chunks / max(total_chunks, 1))

        return report

    def set_fraction(self, page_number: int, fraction: float) -> None:
        with self.lock:
            self.fractions[page_number] = min(max(fraction, self.fractions.get(page_number, 0.0)), 1.0)
            overall = sum(self.fractions.values())
        # Runs on TTS polling threads, so it needs its own session.
        session = SessionLocal()
        try:
            tracked = session.get(Job, self.job_id)
            if tracked and tracked.status == "running":
                tracked.progress = max(tracked.progress or 0, min(100, int(overall / self.total_steps * 100)))
                session.commit()
        finally:
            session.close()


def run_render_video_job(
    job_id: int,
    voice: str | None = None,
    language: str | None = None,
    instruct: str | None = None,
    force_regenerate: bool = False,
) -> None:
    db = SessionLocal()
    tts = TtsService()
    video = VideoService()
    try:
        job = db.get(Job, job_id)
        if not job:
            return
        project = db.get(Project, job.project_id)
        pages = db.query(Page).filter(Page.project_id == job.project_id).order_by(Page.page_number).all()
        if not project or not pages:
            raise RuntimeError("Project pages not found")
        project.status = "rendering"
        update_job(db, job, status="running", progress=1)

        base_dir = project_dir(project.id)
        audio_dir = base_dir / "audio"
        segment_dir = base_dir / "segments"
        total_steps = len(pages) * 2 + 1
        completed = 0
        settings = get_settings()
        tts_progress = TtsProgressAggregator(job_id, total_steps)

        pages_to_synthesize: list[tuple[Page, Path]] = []
        for page in pages:
            existing_audio_path = None if force_regenerate else (safe_storage_path(page.audio_path) if page.audio_path else None)
            if existing_audio_path and existing_audio_path.exists():
                if page.audio_duration is None:
                    page.audio_duration = video.probe_duration(existing_audio_path)
                tts_progress.set_fraction(page.page_number, 1.0)
                completed += 1
            else:
                pages_to_synthesize.append((page, audio_dir / f"page-{page.page_number:04d}.mp3"))

        if pages_to_synthesize:
            tts_workers = min(settings.tts_workers, len(pages_to_synthesize))
            with ThreadPoolExecutor(max_workers=tts_workers) as executor:
                futures = {
                    executor.submit(
                        tts.synthesize,
                        page.transcript,
                        audio_path,
                        voice,
                        language,
                        instruct,
                        progress_callback=tts_progress.reporter(page.page_number),
                    ): (page, audio_path)
                    for page, audio_path in pages_to_synthesize
                }
                for future in as_completed(futures):
                    page, audio_path = futures[future]
                    future.result()
                    page.audio_path = str(audio_path)
                    page.audio_duration = video.probe_duration(audio_path)
                    tts_progress.set_fraction(page.page_number, 1.0)
                    completed += 1
            db.commit()

        render_inputs = [
            (
                page.page_number,
                safe_storage_path(page.image_path),
                safe_storage_path(page.audio_path),
                segment_dir / f"page-{page.page_number:04d}.mp4",
            )
            for page in pages
        ]

        segments_by_page: dict[int, Path] = {}
        max_workers = min(settings.video_segment_workers, len(render_inputs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(video.render_segment, image_path, audio_path, segment_path): page_number
                for page_number, image_path, audio_path, segment_path in render_inputs
            }
            for future in as_completed(futures):
                page_number = futures[future]
                segments_by_page[page_number] = future.result()
                completed += 1
                update_job(db, job, progress=int(completed / total_steps * 100))

        output_path = base_dir / "final.mp4"
        segments = [segments_by_page[page.page_number] for page in pages]
        video.concat_segments(segments, output_path)
        # Caption slots follow the rendered segment durations so cue timing
        # stays aligned with the concatenated video instead of drifting.
        captions = [
            (
                page.transcript,
                page.audio_duration,
                video.probe_duration(segments_by_page[page.page_number]),
                load_timeline(safe_storage_path(page.audio_path)),
            )
            for page in pages
        ]
        video.write_srt(captions, output_path.with_suffix(".srt"))
        project.output_video_path = str(output_path)
        project.status = "completed"
        update_job(db, job, status="completed", progress=100)
    except Exception as exc:
        job = db.get(Job, job_id)
        if job:
            project = db.get(Project, job.project_id)
            if project:
                project.status = "failed"
            update_job(db, job, status="failed", error=str(exc))
    finally:
        db.close()
