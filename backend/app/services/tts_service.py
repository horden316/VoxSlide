from pathlib import Path
from typing import Callable
import base64
import json
import threading
import uuid
from urllib import request
from urllib.error import URLError
from openai import OpenAI
from ..config import get_settings
from .pause_markers import strip_pause_markers

ProgressCallback = Callable[[int, int], None]


def timeline_sidecar_path(audio_path: Path) -> Path:
    """Where the per-chunk timing metadata for an audio file lives."""
    return audio_path.with_suffix(".timeline.json")


class TtsService:
    def available_voices(self) -> list[dict[str, str]]:
        settings = get_settings()
        raw_voices = settings.qwen_tts_voices if settings.tts_provider == "qwen_local" else settings.openai_tts_voice
        voices: list[dict[str, str]] = []
        for raw_voice in raw_voices.split(","):
            value = raw_voice.strip()
            if not value:
                continue
            voice_id, _, label = value.partition(":")
            voice_id = voice_id.strip()
            voices.append({"id": voice_id, "label": label.strip() or voice_id.replace("-", " ").replace("_", " ").title()})
        if not voices:
            default_voice = self.default_voice
            voices.append({"id": default_voice, "label": default_voice})
        return voices

    @property
    def default_voice(self) -> str:
        settings = get_settings()
        return settings.qwen_tts_voice if settings.tts_provider == "qwen_local" else settings.openai_tts_voice

    @property
    def model(self) -> str:
        settings = get_settings()
        return settings.qwen_tts_model if settings.tts_provider == "qwen_local" else settings.openai_tts_model

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str | None = None,
        language: str | None = None,
        instruct: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        settings = get_settings()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected_voice = voice or self.default_voice
        if settings.tts_provider == "qwen_local":
            return self._synthesize_qwen_local(text, output_path, selected_voice, language, instruct, progress_callback)
        return self._synthesize_openai(text, output_path, selected_voice, instruct)

    def _synthesize_openai(self, text: str, output_path: Path, voice: str, instruct: str | None = None) -> Path:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        client = OpenAI(api_key=settings.openai_api_key)
        speech_args = {
            "model": settings.openai_tts_model,
            "voice": voice,
            # OpenAI TTS has no pause-marker support, so drop them instead of reading them aloud.
            "input": strip_pause_markers(text),
            "response_format": "mp3",
        }
        if instruct:
            speech_args["instructions"] = instruct
        with client.audio.speech.with_streaming_response.create(
            **speech_args,
        ) as response:
            response.stream_to_file(output_path)
        return output_path

    def _synthesize_qwen_local(
        self,
        text: str,
        output_path: Path,
        voice: str,
        language: str | None = None,
        instruct: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        settings = get_settings()
        if not settings.qwen_tts_endpoint:
            raise RuntimeError("QWEN_TTS_ENDPOINT is not configured")
        request_id = uuid.uuid4().hex
        request_payload = {
            "text": text,
            "input": text,
            "model": settings.qwen_tts_model,
            "voice": voice,
            "response_format": "mp3",
            "request_id": request_id,
            "include_timeline": True,
        }
        if language:
            request_payload["language"] = language
        if instruct:
            request_payload["instruct"] = instruct
        payload = json.dumps(request_payload).encode("utf-8")
        req = request.Request(
            settings.qwen_tts_endpoint,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "audio/mpeg,application/json"},
            method="POST",
        )
        # Drop any timing metadata from a previous synthesis so a stale sidecar
        # can never describe freshly generated audio.
        timeline_path = timeline_sidecar_path(output_path)
        timeline_path.unlink(missing_ok=True)
        stop_polling = threading.Event()
        poller: threading.Thread | None = None
        if progress_callback:
            progress_url = settings.qwen_tts_endpoint.rsplit("/", 1)[0] + f"/progress/{request_id}"
            poller = threading.Thread(
                target=self._poll_progress,
                args=(progress_url, progress_callback, stop_polling),
                daemon=True,
            )
            poller.start()
        try:
            with request.urlopen(req, timeout=300) as response:
                content_type = response.headers.get("Content-Type", "")
                body = response.read()
        except URLError as exc:
            raise RuntimeError(f"Qwen TTS endpoint is unavailable: {exc}") from exc
        finally:
            stop_polling.set()
            if poller:
                poller.join(timeout=3)

        if "application/json" in content_type:
            self._write_audio_from_json(body, output_path, timeline_path)
        else:
            output_path.write_bytes(body)
        return output_path

    def _poll_progress(self, progress_url: str, callback: ProgressCallback, stop_polling: threading.Event) -> None:
        while not stop_polling.wait(1.0):
            try:
                with request.urlopen(progress_url, timeout=5) as response:
                    state = json.loads(response.read().decode("utf-8"))
            except (URLError, json.JSONDecodeError, ValueError):
                continue
            total = int(state.get("total") or 0)
            if state.get("status") != "running" or total <= 0:
                continue
            try:
                callback(int(state.get("completed") or 0), total)
            except Exception:
                # Progress reporting must never break synthesis.
                continue

    def _write_audio_from_json(self, body: bytes, output_path: Path, timeline_path: Path | None = None) -> None:
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Qwen TTS returned invalid JSON") from exc
        timeline = payload.get("timeline")
        if timeline_path is not None and isinstance(timeline, list) and timeline:
            timeline_path.write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
        audio_base64 = payload.get("audio_base64") or payload.get("audio")
        audio_url = payload.get("audio_url") or payload.get("url")
        if audio_base64:
            if isinstance(audio_base64, str) and "," in audio_base64:
                audio_base64 = audio_base64.split(",", 1)[1]
            output_path.write_bytes(base64.b64decode(audio_base64))
            return
        if audio_url:
            try:
                with request.urlopen(audio_url, timeout=300) as response:
                    output_path.write_bytes(response.read())
            except URLError as exc:
                raise RuntimeError(f"Could not download Qwen TTS audio: {exc}") from exc
            return
        raise RuntimeError("Qwen TTS JSON must include audio_base64, audio, audio_url, or url")
