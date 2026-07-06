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
from fastapi.responses import Response
import numpy as np
from pydantic import BaseModel
import soundfile as sf


MODEL_ID = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEVICE = os.getenv("QWEN_TTS_DEVICE", "cpu")
ATTN_IMPLEMENTATION = os.getenv("QWEN_TTS_ATTN_IMPLEMENTATION", "eager")
DEFAULT_INSTRUCT = os.getenv("QWEN_TTS_INSTRUCT", "Speak in a neutral, consistent, clear voice.")
SEED = int(os.getenv("QWEN_TTS_SEED", "316"))
DO_SAMPLE = os.getenv("QWEN_TTS_DO_SAMPLE", "true").lower() == "true"
TOP_K = int(os.getenv("QWEN_TTS_TOP_K", "50"))
TOP_P = float(os.getenv("QWEN_TTS_TOP_P", "1.0"))
TEMPERATURE = float(os.getenv("QWEN_TTS_TEMPERATURE", "0.7"))
REPETITION_PENALTY = float(os.getenv("QWEN_TTS_REPETITION_PENALTY", "1.05"))
SUBTALKER_DOSAMPLE = os.getenv("QWEN_TTS_SUBTALKER_DOSAMPLE", "true").lower() == "true"
SUBTALKER_TOP_K = int(os.getenv("QWEN_TTS_SUBTALKER_TOP_K", "50"))
SUBTALKER_TOP_P = float(os.getenv("QWEN_TTS_SUBTALKER_TOP_P", "1.0"))
SUBTALKER_TEMPERATURE = float(os.getenv("QWEN_TTS_SUBTALKER_TEMPERATURE", "0.7"))
MAX_CHARS_PER_CHUNK = int(os.getenv("QWEN_TTS_MAX_CHARS_PER_CHUNK", "120"))
MIN_CHUNK_CHARS = int(os.getenv("QWEN_TTS_MIN_CHUNK_CHARS", "6"))
MAX_BATCH_CHUNKS = int(os.getenv("QWEN_TTS_MAX_BATCH_CHUNKS", "3"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_TTS_MAX_NEW_TOKENS", "512"))

SENTENCE_GAP_MS = int(os.getenv("QWEN_TTS_SENTENCE_GAP_MS", "550"))
SEMICOLON_GAP_MS = int(os.getenv("QWEN_TTS_SEMICOLON_GAP_MS", "350"))
PARAGRAPH_GAP_MS = int(os.getenv("QWEN_TTS_PARAGRAPH_GAP_MS", "1000"))
WRAP_GAP_MS = int(os.getenv("QWEN_TTS_WRAP_GAP_MS", "150"))
PAUSE_DEFAULT_MS = int(os.getenv("QWEN_TTS_PAUSE_DEFAULT_MS", "1000"))
TRIM_THRESHOLD_DB = float(os.getenv("QWEN_TTS_TRIM_THRESHOLD_DB", "-42"))
TRIM_PAD_MS = int(os.getenv("QWEN_TTS_TRIM_PAD_MS", "15"))
EDGE_FADE_MS = int(os.getenv("QWEN_TTS_EDGE_FADE_MS", "10"))

logger = logging.getLogger("qwen_tts_service")
logging.basicConfig(level=logging.INFO)

SPEAKERS = {
    "Vivian": {"language": "Chinese", "instruct": "用自然清晰的簡報語氣說"},
    "Serena": {"language": "Chinese", "instruct": "用溫暖柔和的簡報語氣說"},
    "Uncle_Fu": {"language": "Chinese", "instruct": "用沉穩成熟的簡報語氣說"},
    "Dylan": {"language": "Chinese", "instruct": "用年輕自然的北京口音簡報語氣說"},
    "Eric": {"language": "Chinese", "instruct": "用活潑自然的成都口音簡報語氣說"},
    "Ryan": {"language": "English", "instruct": DEFAULT_INSTRUCT},
    "Aiden": {"language": "English", "instruct": DEFAULT_INSTRUCT},
    "Ono_Anna": {"language": "Japanese", "instruct": "自然で聞き取りやすいプレゼン口調で話してください"},
    "Sohee": {"language": "Korean", "instruct": "자연스럽고 또렷한 발표 말투로 말하세요"},
}

PAUSE_MARKER = re.compile(r"\[pause(?::\s*(\d+)\s*(?:ms)?)?\]", re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;.])\s*")
CJK_CHAR = re.compile(r"[　-ヿ㐀-䶿一-鿿가-힯＀-￯]")

app = FastAPI(title="Qwen TTS Service")
model: Any | None = None
generation_lock = Lock()

PROGRESS_TTL_SECONDS = 600
# Progress lives on disk so every uvicorn worker process can answer /progress
# for requests handled by a sibling worker.
PROGRESS_DIR = Path(os.getenv("QWEN_TTS_PROGRESS_DIR", "/tmp/qwen_tts_progress"))
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


@dataclass
class ScriptChunk:
    text: str
    gap_after_ms: int
    explicit_gap: bool = False


def load_model() -> Any:
    global model
    if model is not None:
        return model

    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.float32
    device_map = DEVICE
    if DEVICE.startswith("cuda"):
        dtype = torch.bfloat16

    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    return model


def set_generation_seed() -> None:
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def infer_language(text: str, fallback: str) -> str:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    japanese_count = len(re.findall(r"[\u3040-\u30ff]", text))
    korean_count = len(re.findall(r"[\uac00-\ud7af]", text))
    ascii_letter_count = len(re.findall(r"[A-Za-z]", text))
    if japanese_count > max(cjk_count, korean_count, ascii_letter_count // 2):
        return "Japanese"
    if korean_count > max(cjk_count, japanese_count, ascii_letter_count // 2):
        return "Korean"
    if cjk_count > 0 and cjk_count >= ascii_letter_count // 2:
        return "Chinese"
    if ascii_letter_count > 0:
        return "English"
    return fallback


def join_sentences(left: str, right: str) -> str:
    if not left:
        return right
    if CJK_CHAR.match(left[-1]) and CJK_CHAR.match(right[0]):
        return f"{left}{right}"
    return f"{left} {right}"


def boundary_gap_ms(sentence: str) -> int:
    if sentence and sentence[-1] in "；;":
        return SEMICOLON_GAP_MS
    return SENTENCE_GAP_MS


def parse_segment(segment: str) -> list[ScriptChunk]:
    """Split marker-free text into sentence chunks with punctuation-based gaps."""
    normalized = re.sub(r"\s+", " ", segment).strip()
    if not normalized:
        return []
    sentences = [part.strip() for part in SENTENCE_SPLIT.split(normalized) if part.strip()]
    merged: list[str] = []
    for sentence in sentences:
        if merged and len(merged[-1]) < MIN_CHUNK_CHARS and len(merged[-1]) + len(sentence) <= MAX_CHARS_PER_CHUNK:
            merged[-1] = join_sentences(merged[-1], sentence)
        else:
            merged.append(sentence)
    if len(merged) >= 2 and len(merged[-1]) < MIN_CHUNK_CHARS and len(merged[-2]) + len(merged[-1]) <= MAX_CHARS_PER_CHUNK:
        trailing = merged.pop()
        merged[-1] = join_sentences(merged[-1], trailing)
    chunks: list[ScriptChunk] = []
    for sentence in merged:
        parts = wrap_text(sentence, MAX_CHARS_PER_CHUNK) if len(sentence) > MAX_CHARS_PER_CHUNK else [sentence]
        for part in parts[:-1]:
            chunks.append(ScriptChunk(part, WRAP_GAP_MS))
        chunks.append(ScriptChunk(parts[-1], boundary_gap_ms(sentence)))
    return chunks


def parse_paragraph(paragraph: str) -> list[ScriptChunk]:
    chunks: list[ScriptChunk] = []
    position = 0
    for match in PAUSE_MARKER.finditer(paragraph):
        pause_ms = int(match.group(1)) if match.group(1) else PAUSE_DEFAULT_MS
        segment_chunks = parse_segment(paragraph[position : match.start()])
        if segment_chunks:
            segment_chunks[-1].gap_after_ms = pause_ms
            segment_chunks[-1].explicit_gap = True
            chunks.extend(segment_chunks)
        elif chunks:
            chunks[-1].gap_after_ms += pause_ms
            chunks[-1].explicit_gap = True
        position = match.end()
    chunks.extend(parse_segment(paragraph[position:]))
    return chunks


def parse_script(text: str) -> list[ScriptChunk]:
    chunks: list[ScriptChunk] = []
    for paragraph in re.split(r"\n\s*\n+", text.strip()):
        paragraph_chunks = parse_paragraph(paragraph)
        if not paragraph_chunks:
            continue
        if chunks and not chunks[-1].explicit_gap:
            chunks[-1].gap_after_ms = PARAGRAPH_GAP_MS
        chunks.extend(paragraph_chunks)
    if chunks and not chunks[-1].explicit_gap:
        chunks[-1].gap_after_ms = 0
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


def chunked(items: list[str], size: int) -> list[list[str]]:
    batch_size = max(size, 1)
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def trim_edges(wav: np.ndarray, sr: int) -> np.ndarray:
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
    threshold = peak * (10 ** (TRIM_THRESHOLD_DB / 20))
    rms = np.sqrt(np.mean(np.square(wav[: frames * frame_len].reshape(frames, frame_len)), axis=1))
    active = np.nonzero(rms > threshold)[0]
    if active.size == 0:
        return wav[:0]
    pad = int(sr * TRIM_PAD_MS / 1000)
    start = max(active[0] * frame_len - pad, 0)
    end = min((active[-1] + 1) * frame_len + pad, len(wav))
    return wav[start:end]


def apply_edge_fades(wav: np.ndarray, sr: int) -> np.ndarray:
    fade_len = min(int(sr * EDGE_FADE_MS / 1000), len(wav) // 2)
    if fade_len <= 0:
        return wav
    wav = wav.copy()
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=wav.dtype)
    wav[:fade_len] *= ramp
    wav[-fade_len:] *= ramp[::-1]
    return wav


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


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

    speaker = payload.speaker or payload.voice or "Ryan"
    speaker_config = SPEAKERS.get(speaker)
    if not speaker_config:
        raise HTTPException(status_code=400, detail=f"Unsupported speaker: {speaker}")

    script_chunks = parse_script(text)
    if not script_chunks:
        raise HTTPException(status_code=400, detail="text contains no speakable content")

    set_progress(payload.request_id, 0, len(script_chunks), "queued")
    try:
        spoken_text = PAUSE_MARKER.sub(" ", text)
        language = payload.language or infer_language(spoken_text, speaker_config["language"])
        chunk_texts = [chunk.text for chunk in script_chunks]
        logger.info(
            "Queueing TTS speaker=%s language=%s chars=%s chunks=%s seed=%s do_sample=%s subtalker_dosample=%s",
            speaker,
            language,
            len(text),
            len(chunk_texts),
            SEED,
            DO_SAMPLE,
            SUBTALKER_DOSAMPLE,
        )
        with generation_lock:
            started_at = time.perf_counter()
            qwen_model = load_model()
            set_generation_seed()
            audio_chunks = []
            sr = None
            completed_chunks = 0
            set_progress(payload.request_id, 0, len(script_chunks), "running")
            for batch_index, batch in enumerate(chunked(chunk_texts, MAX_BATCH_CHUNKS), start=1):
                batch_started_at = time.perf_counter()
                wavs, chunk_sr = qwen_model.generate_custom_voice(
                    text=batch,
                    language=[language] * len(batch),
                    speaker=[speaker] * len(batch),
                    instruct=[payload.instruct or speaker_config["instruct"]] * len(batch),
                    do_sample=DO_SAMPLE,
                    top_k=TOP_K,
                    top_p=TOP_P,
                    temperature=TEMPERATURE,
                    repetition_penalty=REPETITION_PENALTY,
                    max_new_tokens=MAX_NEW_TOKENS,
                    subtalker_dosample=SUBTALKER_DOSAMPLE,
                    subtalker_top_k=SUBTALKER_TOP_K,
                    subtalker_top_p=SUBTALKER_TOP_P,
                    subtalker_temperature=SUBTALKER_TEMPERATURE,
                )
                sr = chunk_sr
                audio_chunks.extend(wavs)
                completed_chunks += len(batch)
                set_progress(payload.request_id, completed_chunks, len(script_chunks), "running")
                logger.info(
                    "Generated TTS chunk batch %s speaker=%s chunks=%s/%s chars=%s elapsed=%.2fs",
                    batch_index,
                    speaker,
                    completed_chunks,
                    len(chunk_texts),
                    sum(len(chunk) for chunk in batch),
                    time.perf_counter() - batch_started_at,
                )
            sample_rate = sr or 24000
            wav_parts: list[np.ndarray] = []
            for chunk_meta, audio_chunk in zip(script_chunks, audio_chunks):
                processed = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
                processed = apply_edge_fades(trim_edges(processed, sample_rate), sample_rate)
                wav_parts.append(processed)
                if chunk_meta.gap_after_ms > 0:
                    wav_parts.append(np.zeros(int(sample_rate * chunk_meta.gap_after_ms / 1000), dtype=np.float32))
            wav = np.concatenate(wav_parts) if wav_parts else np.array([], dtype=np.float32)
            logger.info(
                "Generated TTS speaker=%s chars=%s chunks=%s elapsed=%.2fs",
                speaker,
                len(text),
                len(chunk_texts),
                time.perf_counter() - started_at,
            )
    except Exception as exc:
        set_progress(payload.request_id, 0, len(script_chunks), "failed")
        raise HTTPException(status_code=500, detail=f"Qwen TTS generation failed: {exc}") from exc

    set_progress(payload.request_id, len(script_chunks), len(script_chunks), "completed")
    output = BytesIO()
    sf.write(output, wav, sr or 24000, format="MP3")
    return Response(content=output.getvalue(), media_type="audio/mpeg")
