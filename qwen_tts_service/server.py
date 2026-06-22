from io import BytesIO
import logging
import os
import random
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import numpy as np
from pydantic import BaseModel
import soundfile as sf


MODEL_ID = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
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

logger = logging.getLogger("qwen_tts_service")

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
        qwen_model = load_model()
        set_generation_seed()
        logger.info(
            "Generating TTS speaker=%s language=%s chars=%s seed=%s do_sample=%s subtalker_dosample=%s",
            speaker,
            payload.language or speaker_config["language"],
            len(text),
            SEED,
            DO_SAMPLE,
            SUBTALKER_DOSAMPLE,
        )
        wavs, sr = qwen_model.generate_custom_voice(
            text=text,
            language=payload.language or speaker_config["language"],
            speaker=speaker,
            instruct=payload.instruct or speaker_config["instruct"],
            do_sample=DO_SAMPLE,
            top_k=TOP_K,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            subtalker_dosample=SUBTALKER_DOSAMPLE,
            subtalker_top_k=SUBTALKER_TOP_K,
            subtalker_top_p=SUBTALKER_TOP_P,
            subtalker_temperature=SUBTALKER_TEMPERATURE,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qwen TTS generation failed: {exc}") from exc

    output = BytesIO()
    sf.write(output, wavs[0], sr, format="MP3")
    return Response(content=output.getvalue(), media_type="audio/mpeg")
