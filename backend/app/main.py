from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .config import get_settings
from .database import init_db
from .routes import jobs, pages, projects
from .storage import safe_storage_path


settings = get_settings()
app = FastAPI(title="VoxSlide API")

origins = [origin.strip() for origin in settings.backend_cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/files/{file_path:path}")
def serve_file(file_path: str) -> FileResponse:
    path = safe_storage_path(settings.storage_dir / Path(file_path))
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


app.include_router(projects.router)
app.include_router(pages.router)
app.include_router(jobs.router)
