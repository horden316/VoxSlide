# VoxSlide

Dockerized MVP for turning a presentation PDF plus per-page transcripts into a narrated MP4 video.

## Stack

- Frontend: Next.js, React, TypeScript, Tailwind CSS
- Backend: FastAPI, SQLite, SQLAlchemy
- Media: PyMuPDF for PDF rendering, OpenAI TTS, FFmpeg/ffprobe for video assembly

## Setup

```bash
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` before generating audio or rendering videos. Do not commit real keys.

## Run

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs

Uploaded PDFs, rendered page images, audio, SQLite DB, and final videos are stored in `./storage`.

## Manual Test

1. Open http://localhost:3000.
2. Create a project.
3. Upload a PDF.
4. Confirm page previews appear.
5. Enter transcripts and click save on each page.
6. Click generate audio on one page and play it.
7. Click render full video.
8. Watch the progress indicator until completion.
9. Download `final.mp4`.

## API

- `POST /api/projects`
- `GET /api/projects`
- `GET /api/projects/{project_id}`
- `POST /api/projects/{project_id}/upload-pdf`
- `GET /api/projects/{project_id}/pages`
- `PATCH /api/pages/{page_id}`
- `POST /api/pages/{page_id}/generate-audio`
- `POST /api/projects/{project_id}/render-video`
- `GET /api/jobs/{job_id}`
- `GET /api/projects/{project_id}/download`

## Local Backend Smoke Check

With Docker running:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```
