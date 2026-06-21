from pathlib import Path
import base64
import json
from urllib import request
from urllib.error import URLError
from openai import OpenAI
from ..config import get_settings


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

    def synthesize(self, text: str, output_path: Path, voice: str | None = None) -> Path:
        settings = get_settings()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected_voice = voice or self.default_voice
        if settings.tts_provider == "qwen_local":
            return self._synthesize_qwen_local(text, output_path, selected_voice)
        return self._synthesize_openai(text, output_path, selected_voice)

    def _synthesize_openai(self, text: str, output_path: Path, voice: str) -> Path:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        client = OpenAI(api_key=settings.openai_api_key)
        with client.audio.speech.with_streaming_response.create(
            model=settings.openai_tts_model,
            voice=voice,
            input=text,
            response_format="mp3",
        ) as response:
            response.stream_to_file(output_path)
        return output_path

    def _synthesize_qwen_local(self, text: str, output_path: Path, voice: str) -> Path:
        settings = get_settings()
        if not settings.qwen_tts_endpoint:
            raise RuntimeError("QWEN_TTS_ENDPOINT is not configured")
        payload = json.dumps(
            {
                "text": text,
                "input": text,
                "model": settings.qwen_tts_model,
                "voice": voice,
                "response_format": "mp3",
            }
        ).encode("utf-8")
        req = request.Request(
            settings.qwen_tts_endpoint,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "audio/mpeg,application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=300) as response:
                content_type = response.headers.get("Content-Type", "")
                body = response.read()
        except URLError as exc:
            raise RuntimeError(f"Qwen TTS endpoint is unavailable: {exc}") from exc

        if "application/json" in content_type:
            self._write_audio_from_json(body, output_path)
        else:
            output_path.write_bytes(body)
        return output_path

    def _write_audio_from_json(self, body: bytes, output_path: Path) -> None:
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Qwen TTS returned invalid JSON") from exc
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
