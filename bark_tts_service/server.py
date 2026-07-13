import base64
from dataclasses import dataclass
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import random
import re
from threading import Lock
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
import numpy as np
from pydantic import BaseModel, ValidationError
import soundfile as sf


MODEL_ID = os.getenv("BARK_TTS_MODEL", "suno/bark")
DEVICE = os.getenv("BARK_TTS_DEVICE", "cuda:0")
SEED = int(os.getenv("BARK_TTS_SEED", "316"))
DO_SAMPLE = os.getenv("BARK_TTS_DO_SAMPLE", "true").lower() == "true"
# Bark's community-settled recipe: moderate semantic/coarse sampling with a
# cooler fine layer keeps prosody varied without garbling articulation.
SEMANTIC_TEMPERATURE = float(os.getenv("BARK_TTS_SEMANTIC_TEMPERATURE", "0.7"))
COARSE_TEMPERATURE = float(os.getenv("BARK_TTS_COARSE_TEMPERATURE", "0.7"))
FINE_TEMPERATURE = float(os.getenv("BARK_TTS_FINE_TEMPERATURE", "0.5"))
# Bark degrades past ~13-14s of speech per generation, so chunks stay at one
# sentence where possible and hard-wrap around ~220 chars.
MAX_CHARS_PER_CHUNK = int(os.getenv("BARK_TTS_MAX_CHARS_PER_CHUNK", "220"))
MIN_CHUNK_CHARS = int(os.getenv("BARK_TTS_MIN_CHUNK_CHARS", "0"))

SENTENCE_GAP_MS = int(os.getenv("BARK_TTS_SENTENCE_GAP_MS", "700"))
SEMICOLON_GAP_MS = int(os.getenv("BARK_TTS_SEMICOLON_GAP_MS", "350"))
PARAGRAPH_GAP_MS = int(os.getenv("BARK_TTS_PARAGRAPH_GAP_MS", "1000"))
WRAP_GAP_MS = int(os.getenv("BARK_TTS_WRAP_GAP_MS", "150"))
PAUSE_DEFAULT_MS = int(os.getenv("BARK_TTS_PAUSE_DEFAULT_MS", "1000"))
TRIM_THRESHOLD_DB = float(os.getenv("BARK_TTS_TRIM_THRESHOLD_DB", "-42"))
TRIM_PAD_MS = int(os.getenv("BARK_TTS_TRIM_PAD_MS", "15"))
EDGE_FADE_MS = int(os.getenv("BARK_TTS_EDGE_FADE_MS", "10"))

FALLBACK_SAMPLE_RATE = 24000

logger = logging.getLogger("bark_tts_service")
logging.basicConfig(level=logging.INFO)

PAUSE_MARKER = re.compile(r"\[pause(?::\s*(\d+)\s*(?:ms)?)?\]", re.IGNORECASE)
# Keep in sync with PRONUNCIATION_MARKER in backend/app/services/pronunciation_markers.py.
# [dis:display text|read:spoken text] — synthesis reads the read side while the
# timeline reports the dis side, so subtitles show the original spelling.
PRONUNCIATION_MARKER = re.compile(r"\[dis:([^|\[\]]*)\|read:([^\[\]]*)\]", re.IGNORECASE)
# Each pronunciation marker is swapped for one private-use character before
# chunking so sentence splitting and wrapping can never cut a marker apart.
PRONUNCIATION_PLACEHOLDER_BASE = 0xE000
PRONUNCIATION_PLACEHOLDER = re.compile("[-]")
QUOTE_CHARS = re.compile(r"[\"“”„‟«»「」『』＂]")
SENTENCE_SPLIT = re.compile(r"(?<=[。！？；])\s*|(?<=[!?;.])\s+")
CJK_CHAR = re.compile(r"[　-ヿ㐀-䶿一-鿿가-힯＀-￯]")
VOICE_PATTERN = re.compile(r"^v2/[a-z]{2}_speaker_[0-9]$")

app = FastAPI(title="Bark TTS Service")
processor: Any | None = None
model: Any | None = None
generation_lock = Lock()

PROGRESS_TTL_SECONDS = 600
# Progress lives on disk so every uvicorn worker process can answer /progress
# for requests handled by a sibling worker.
PROGRESS_DIR = Path(os.getenv("BARK_TTS_PROGRESS_DIR", "/tmp/bark_tts_progress"))
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def progress_path(request_id: str) -> Path | None:
    if not REQUEST_ID_PATTERN.match(request_id):
        return None
    return PROGRESS_DIR / f"{request_id}.json"


def set_progress(request_id: str | None, completed: int, total: int, status: str) -> None:
    if not request_id:
        return
    path = progress_path(request_id)
    if path is None:
        return
    now = time.time()
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps({"completed": completed, "total": total, "status": status}))
    tmp_path.replace(path)
    for stale in PROGRESS_DIR.glob("*.json"):
        try:
            if now - stale.stat().st_mtime > PROGRESS_TTL_SECONDS:
                stale.unlink(missing_ok=True)
        except OSError:
            continue


class TtsParams(BaseModel):
    """Per-request tuning knobs; every field falls back to its env-configured default."""

    model_config = {"extra": "ignore"}

    seed: int = SEED
    do_sample: bool = DO_SAMPLE
    semantic_temperature: float = SEMANTIC_TEMPERATURE
    coarse_temperature: float = COARSE_TEMPERATURE
    fine_temperature: float = FINE_TEMPERATURE
    max_chars_per_chunk: int = MAX_CHARS_PER_CHUNK
    min_chunk_chars: int = MIN_CHUNK_CHARS
    sentence_gap_ms: int = SENTENCE_GAP_MS
    semicolon_gap_ms: int = SEMICOLON_GAP_MS
    paragraph_gap_ms: int = PARAGRAPH_GAP_MS
    wrap_gap_ms: int = WRAP_GAP_MS
    pause_default_ms: int = PAUSE_DEFAULT_MS
    trim_threshold_db: float = TRIM_THRESHOLD_DB
    trim_pad_ms: int = TRIM_PAD_MS
    edge_fade_ms: int = EDGE_FADE_MS


class TtsRequest(BaseModel):
    text: str | None = None
    input: str | None = None
    model: str | None = None
    voice: str | None = None
    speaker: str | None = None
    language: str | None = None
    instruct: str | None = None
    response_format: str | None = "mp3"
    request_id: str | None = None
    include_timeline: bool = False
    params: dict[str, Any] | None = None


@dataclass
class ScriptChunk:
    text: str
    gap_after_ms: int
    explicit_gap: bool = False
    display_text: str = ""


def extract_pronunciation_markers(text: str) -> tuple[str, list[tuple[str, str]]]:
    pronunciations: list[tuple[str, str]] = []

    def replace(match: re.Match) -> str:
        display = match.group(1).strip()
        spoken = match.group(2).strip() or display
        pronunciations.append((display, spoken))
        return chr(PRONUNCIATION_PLACEHOLDER_BASE + len(pronunciations) - 1)

    return PRONUNCIATION_MARKER.sub(replace, text), pronunciations


def expand_pronunciations(text: str, pronunciations: list[tuple[str, str]], spoken: bool) -> str:
    def replace(match: re.Match) -> str:
        index = ord(match.group(0)) - PRONUNCIATION_PLACEHOLDER_BASE
        if 0 <= index < len(pronunciations):
            display, read = pronunciations[index]
            return read if spoken else display
        return ""

    return PRONUNCIATION_PLACEHOLDER.sub(replace, text)


def load_model() -> tuple[Any, Any]:
    global processor, model
    if processor is not None and model is not None:
        return processor, model

    import torch
    from transformers import AutoProcessor, BarkModel

    dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = BarkModel.from_pretrained(MODEL_ID, dtype=dtype).to(DEVICE)
    model.eval()
    return processor, model


def sample_rate_of(bark_model: Any) -> int:
    return int(getattr(bark_model.generation_config, "sample_rate", FALLBACK_SAMPLE_RATE) or FALLBACK_SAMPLE_RATE)


def set_generation_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def join_sentences(left: str, right: str) -> str:
    if not left:
        return right
    if CJK_CHAR.match(left[-1]) and CJK_CHAR.match(right[0]):
        return f"{left}{right}"
    return f"{left} {right}"


def boundary_gap_ms(sentence: str, params: TtsParams) -> int:
    if sentence and sentence[-1] in "；;":
        return params.semicolon_gap_ms
    return params.sentence_gap_ms


def parse_segment(segment: str, params: TtsParams) -> list[ScriptChunk]:
    """Split marker-free text into sentence chunks with punctuation-based gaps."""
    normalized = re.sub(r"\s+", " ", QUOTE_CHARS.sub("", segment)).strip()
    if not normalized:
        return []
    sentences = [part.strip() for part in SENTENCE_SPLIT.split(normalized) if part.strip()]
    merged: list[str] = []
    for sentence in sentences:
        if merged and len(merged[-1]) < params.min_chunk_chars and len(merged[-1]) + len(sentence) <= params.max_chars_per_chunk:
            merged[-1] = join_sentences(merged[-1], sentence)
        else:
            merged.append(sentence)
    if len(merged) >= 2 and len(merged[-1]) < params.min_chunk_chars and len(merged[-2]) + len(merged[-1]) <= params.max_chars_per_chunk:
        trailing = merged.pop()
        merged[-1] = join_sentences(merged[-1], trailing)
    chunks: list[ScriptChunk] = []
    for sentence in merged:
        parts = wrap_text(sentence, params.max_chars_per_chunk) if len(sentence) > params.max_chars_per_chunk else [sentence]
        for part in parts[:-1]:
            chunks.append(ScriptChunk(part, params.wrap_gap_ms))
        chunks.append(ScriptChunk(parts[-1], boundary_gap_ms(sentence, params)))
    return chunks


def parse_paragraph(paragraph: str, params: TtsParams) -> list[ScriptChunk]:
    chunks: list[ScriptChunk] = []
    position = 0
    for match in PAUSE_MARKER.finditer(paragraph):
        pause_ms = int(match.group(1)) if match.group(1) else params.pause_default_ms
        segment_chunks = parse_segment(paragraph[position : match.start()], params)
        if segment_chunks:
            segment_chunks[-1].gap_after_ms = pause_ms
            segment_chunks[-1].explicit_gap = True
            chunks.extend(segment_chunks)
        elif chunks:
            chunks[-1].gap_after_ms += pause_ms
            chunks[-1].explicit_gap = True
        position = match.end()
    chunks.extend(parse_segment(paragraph[position:], params))
    return chunks


def parse_script(text: str, params: TtsParams) -> list[ScriptChunk]:
    protected, pronunciations = extract_pronunciation_markers(text)
    chunks: list[ScriptChunk] = []
    for paragraph in re.split(r"\n\s*\n+", protected.strip()):
        paragraph_chunks = parse_paragraph(paragraph, params)
        if not paragraph_chunks:
            continue
        if chunks and not chunks[-1].explicit_gap:
            chunks[-1].gap_after_ms = params.paragraph_gap_ms
        chunks.extend(paragraph_chunks)
    if chunks and not chunks[-1].explicit_gap:
        chunks[-1].gap_after_ms = 0
    for chunk in chunks:
        placeholder_text = chunk.text
        chunk.text = expand_pronunciations(placeholder_text, pronunciations, spoken=True)
        chunk.display_text = expand_pronunciations(placeholder_text, pronunciations, spoken=False)
    return chunks


def wrap_text(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        split_at = find_split_position(remaining, max_chars)
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def find_split_position(text: str, max_chars: int) -> int:
    window = text[: max_chars + 1]
    for pattern in [r"[,，、:：]\s*", r"\s+"]:
        matches = list(re.finditer(pattern, window))
        if matches:
            return matches[-1].end()
    return max_chars


def trim_edges(wav: np.ndarray, sr: int, params: TtsParams) -> np.ndarray:
    """Cut the model's variable leading/trailing silence so inserted gaps stay exact."""
    if wav.size == 0:
        return wav
    frame_len = max(int(sr * 0.01), 1)
    frames = len(wav) // frame_len
    if frames == 0:
        return wav
    peak = float(np.max(np.abs(wav)))
    if peak <= 0.0:
        return wav[:0]
    threshold = peak * (10 ** (params.trim_threshold_db / 20))
    rms = np.sqrt(np.mean(np.square(wav[: frames * frame_len].reshape(frames, frame_len)), axis=1))
    active = np.nonzero(rms > threshold)[0]
    if active.size == 0:
        return wav[:0]
    pad = int(sr * params.trim_pad_ms / 1000)
    start = max(active[0] * frame_len - pad, 0)
    end = min((active[-1] + 1) * frame_len + pad, len(wav))
    return wav[start:end]


def apply_edge_fades(wav: np.ndarray, sr: int, params: TtsParams) -> np.ndarray:
    fade_len = min(int(sr * params.edge_fade_ms / 1000), len(wav) // 2)
    if fade_len <= 0:
        return wav
    wav = wav.copy()
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=wav.dtype)
    wav[:fade_len] *= ramp
    wav[-fade_len:] *= ramp[::-1]
    return wav


def synthesize_chunk(bark_processor: Any, bark_model: Any, text: str, voice: str, params: TtsParams) -> np.ndarray:
    import torch

    inputs = bark_processor(text, voice_preset=voice, return_tensors="pt").to(bark_model.device)
    with torch.inference_mode():
        audio = bark_model.generate(
            **inputs,
            do_sample=params.do_sample,
            semantic_temperature=params.semantic_temperature,
            coarse_temperature=params.coarse_temperature,
            fine_temperature=params.fine_temperature,
        )
    return audio.squeeze().float().cpu().numpy().reshape(-1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


@app.get("/params")
def default_params() -> dict[str, Any]:
    """Expose the env-configured defaults so clients can render tuning UIs."""
    return TtsParams().model_dump()


@app.get("/progress/{request_id}")
def get_progress(request_id: str) -> dict[str, Any]:
    path = progress_path(request_id)
    if path is not None and path.exists():
        try:
            state = json.loads(path.read_text())
            return {"completed": state["completed"], "total": state["total"], "status": state["status"]}
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    return {"completed": 0, "total": 0, "status": "unknown"}


@app.post("/tts")
def tts(payload: TtsRequest) -> Response:
    text = (payload.text or payload.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text or input is required")

    voice = payload.speaker or payload.voice or "v2/en_speaker_6"
    if not VOICE_PATTERN.match(voice):
        raise HTTPException(status_code=400, detail=f"Unsupported voice preset: {voice}")

    try:
        params = TtsParams(**(payload.params or {}))
    except (ValidationError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid TTS params: {exc}") from exc

    script_chunks = parse_script(text, params)
    if not script_chunks:
        raise HTTPException(status_code=400, detail="text contains no speakable content")

    set_progress(payload.request_id, 0, len(script_chunks), "queued")
    try:
        logger.info(
            "Queueing TTS voice=%s chars=%s chunks=%s seed=%s do_sample=%s",
            voice,
            len(text),
            len(script_chunks),
            params.seed,
            params.do_sample,
        )
        with generation_lock:
            started_at = time.perf_counter()
            bark_processor, bark_model = load_model()
            sample_rate = sample_rate_of(bark_model)
            # Seeded like the Qwen service: the same seed reproduces a take and
            # a rerolled seed produces a fresh one when Bark misbehaves.
            set_generation_seed(params.seed)
            set_progress(payload.request_id, 0, len(script_chunks), "running")
            wav_parts: list[np.ndarray] = []
            timeline: list[dict[str, Any]] = []
            cursor_samples = 0
            for index, chunk_meta in enumerate(script_chunks, start=1):
                chunk_started_at = time.perf_counter()
                raw = synthesize_chunk(bark_processor, bark_model, chunk_meta.text, voice, params)
                processed = apply_edge_fades(trim_edges(raw, sample_rate, params), sample_rate, params)
                wav_parts.append(processed)
                timeline.append(
                    {
                        "text": chunk_meta.display_text or chunk_meta.text,
                        "start": round(cursor_samples / sample_rate, 3),
                        "end": round((cursor_samples + len(processed)) / sample_rate, 3),
                    }
                )
                cursor_samples += len(processed)
                if chunk_meta.gap_after_ms > 0:
                    gap_samples = int(sample_rate * chunk_meta.gap_after_ms / 1000)
                    wav_parts.append(np.zeros(gap_samples, dtype=np.float32))
                    cursor_samples += gap_samples
                set_progress(payload.request_id, index, len(script_chunks), "running")
                logger.info(
                    "Generated TTS chunk %s/%s voice=%s chars=%s elapsed=%.2fs",
                    index,
                    len(script_chunks),
                    voice,
                    len(chunk_meta.text),
                    time.perf_counter() - chunk_started_at,
                )
            wav = np.concatenate(wav_parts) if wav_parts else np.array([], dtype=np.float32)
            logger.info(
                "Generated TTS voice=%s chars=%s chunks=%s elapsed=%.2fs",
                voice,
                len(text),
                len(script_chunks),
                time.perf_counter() - started_at,
            )
    except Exception as exc:
        set_progress(payload.request_id, 0, len(script_chunks), "failed")
        raise HTTPException(status_code=500, detail=f"Bark TTS generation failed: {exc}") from exc

    set_progress(payload.request_id, len(script_chunks), len(script_chunks), "completed")
    output = BytesIO()
    sf.write(output, wav, sample_rate, format="MP3")
    audio_bytes = output.getvalue()
    if payload.include_timeline:
        return JSONResponse(
            {
                "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                "timeline": timeline,
                "duration": round(len(wav) / sample_rate, 3),
            }
        )
    return Response(content=audio_bytes, media_type="audio/mpeg")
