from pathlib import Path
from uuid import uuid4
from fastapi import HTTPException
from .config import get_settings


settings = get_settings()


def project_dir(project_id: int) -> Path:
    path = settings.storage_dir / "projects" / str(project_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_storage_path(path: str | Path) -> Path:
    base = settings.storage_dir.resolve()
    candidate = Path(path).resolve()
    if base != candidate and base not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid storage path")
    return candidate


def unique_filename(suffix: str) -> str:
    return f"{uuid4().hex}{suffix.lower()}"


def public_file_url(path: str | None) -> str | None:
    if not path:
        return None
    safe = safe_storage_path(path)
    relative = safe.relative_to(settings.storage_dir.resolve())
    return f"/api/files/{relative.as_posix()}"
