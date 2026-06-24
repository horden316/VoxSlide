from io import BytesIO
import logging
import os
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
DO_SAMPLE = os.getenv("QWEN_TTS_DO_SAMPLE", "false").lower() == "true"
TOP_K = int(os.getenv("QWEN_TTS_TOP_K", "50"))
TOP_P = float(os.getenv("QWEN_TTS_TOP_P", "1.0"))
TEMPERATURE = float(os.getenv("QWEN_TTS_TEMPERATURE", "0.7"))
REPETITION_PENALTY = float(os.getenv("QWEN_TTS_REPETITION_PENALTY", "1.05"))
SUBTALKER_DOSAMPLE = os.getenv("QWEN_TTS_SUBTALKER_DOSAMPLE", "false").lower() == "true"
SUBTALKER_TOP_K = int(os.getenv("QWEN_TTS_SUBTALKER_TOP_K", "50"))
SUBTALKER_TOP_P = float(os.getenv("QWEN_TTS_SUBTALKER_TOP_P", "1.0"))
SUBTALKER_TEMPERATURE = float(os.getenv("QWEN_TTS_SUBTALKER_TEMPERATURE", "0.7"))
MAX_CHARS_PER_CHUNK = int(os.getenv("QWEN_TTS_MAX_CHARS_PER_CHUNK", "120"))
CHUNK_SILENCE_MS = int(os.getenv("QWEN_TTS_CHUNK_SILENCE_MS", "300"))
MAX_BATCH_CHUNKS = int(os.getenv("QWEN_TTS_MAX_BATCH_CHUNKS", "3"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_TTS_MAX_NEW_TOKENS", "128"))

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

app = FastAPI(title="Qwen TTS Service")
model: Any | None = None
generation_lock = Lock()


class TtsRequest(BaseModel):
    text: str | None = None
    input: str | None = None
    model: str | None = None
    voice: str | None = None
    speaker: str | None = None
    language: str | None = None
    instruct: str | None = None
    response_format: str | None = "mp3"


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


def split_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?；;.!?])\s*", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(wrap_text(sentence, max_chars))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/tts")
def tts(payload: TtsRequest) -> Response:
    text = (payload.text or payload.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text or input is required")

    speaker = payload.speaker or payload.voice or "Ryan"
    speaker_config = SPEAKERS.get(speaker)
    if not speaker_config:
        raise HTTPException(status_code=400, detail=f"Unsupported speaker: {speaker}")

    try:
        language = payload.language or infer_language(text, speaker_config["language"])
        chunks = split_text(text)
        logger.info(
            "Queueing TTS speaker=%s language=%s chars=%s seed=%s do_sample=%s subtalker_dosample=%s",
            speaker,
            language,
            len(text),
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
            for batch_index, batch in enumerate(chunked(chunks, MAX_BATCH_CHUNKS), start=1):
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
                logger.info(
                    "Generated TTS chunk batch %s speaker=%s chunks=%s/%s chars=%s elapsed=%.2fs",
                    batch_index,
                    speaker,
                    completed_chunks,
                    len(chunks),
                    sum(len(chunk) for chunk in batch),
                    time.perf_counter() - batch_started_at,
                )
            silence = np.zeros(int((sr or 24000) * CHUNK_SILENCE_MS / 1000), dtype=np.float32)
            wav_parts: list[np.ndarray] = []
            for index, chunk in enumerate(audio_chunks):
                wav_parts.append(chunk)
                if index < len(audio_chunks) - 1:
                    wav_parts.append(silence)
            wav = np.concatenate(wav_parts) if wav_parts else np.array([], dtype=np.float32)
            logger.info("Generated TTS speaker=%s chars=%s chunks=%s elapsed=%.2fs", speaker, len(text), len(chunks), time.perf_counter() - started_at)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qwen TTS generation failed: {exc}") from exc

    output = BytesIO()
    sf.write(output, wav, sr or 24000, format="MP3")
    return Response(content=output.getvalue(), media_type="audio/mpeg")
