# VoxSlide

Dockerized solution for turning a presentation PDF plus per-page transcripts into a narrated MP4 video.

## Stack

- Frontend: Next.js, React, TypeScript, Tailwind CSS
- Backend: FastAPI, SQLite, SQLAlchemy
- Media: PyMuPDF for PDF rendering, local Qwen TTS or OpenAI TTS, FFmpeg/ffprobe for video assembly

## Setup

```bash
cp .env.example .env
```

By default this branch runs a local `qwen-tts` service using `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, the largest CustomVoice checkpoint in the Qwen3-TTS collection. The backend calls `QWEN_TTS_ENDPOINT` with a JSON `POST` containing `text`, `input`, `model`, `voice`, and `response_format`, then stores the returned MP3 bytes.

The bundled Qwen service exposes the CustomVoice speakers `Ryan`, `Aiden`, `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ono_Anna`, and `Sohee`. On Apple Silicon Docker this runs on CPU by default, so first audio generation can be slow while the model downloads and loads.

To use OpenAI TTS instead, set `TTS_PROVIDER=openai` and set `OPENAI_API_KEY`. Do not commit real keys.

A local `bark-tts` service (Suno Bark via `transformers`) is also bundled; select the Bark provider in the UI or set `TTS_PROVIDER=bark_local`. Bark's sampling makes each take different, so the per-page Reroll button (shared with Qwen) regenerates with a fresh seed until a take sounds right.

## Run

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs

When opening the frontend from another machine, use the server host, for example
`http://172.17.14.8:3000`. The frontend will rewrite the default
`http://localhost:8000` API URL to the same host on port `8000` in the browser.

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
