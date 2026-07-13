from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable
import base64
import json
import subprocess
import tempfile
import threading
import uuid
from urllib import request
from urllib.error import URLError
from openai import OpenAI
from ..config import get_settings
from .script_chunks import ScriptChunk, parse_script

ProgressCallback = Callable[[int, int], None]

VALID_PROVIDERS = ("qwen_local", "kokoro_local", "bark_local", "chatterbox_local", "openai")


def timeline_sidecar_path(audio_path: Path) -> Path:
    """Where the per-chunk timing metadata for an audio file lives."""
    return audio_path.with_suffix(".timeline.json")


class TtsService:
    def resolve_provider(self, provider: str | None = None) -> str:
        value = (provider or get_settings().tts_provider or "").strip()
        if value not in VALID_PROVIDERS:
            raise RuntimeError(f"Unknown TTS provider: {value!r} (expected one of {', '.join(VALID_PROVIDERS)})")
        return value

    def available_voices(self, provider: str | None = None) -> list[dict[str, str]]:
        settings = get_settings()
        raw_voices = {
            "qwen_local": settings.qwen_tts_voices,
            "kokoro_local": settings.kokoro_tts_voices,
            "bark_local": settings.bark_tts_voices,
            "chatterbox_local": settings.chatterbox_tts_voices,
            "openai": settings.openai_tts_voices,
        }[self.resolve_provider(provider)]
        voices: list[dict[str, str]] = []
        for raw_voice in raw_voices.split(","):
            value = raw_voice.strip()
            if not value:
                continue
            voice_id, _, label = value.partition(":")
            voice_id = voice_id.strip()
            voices.append({"id": voice_id, "label": label.strip() or voice_id.replace("-", " ").replace("_", " ").title()})
        if not voices:
            default_voice = self.default_voice(provider)
            voices.append({"id": default_voice, "label": default_voice})
        return voices

    def default_voice(self, provider: str | None = None) -> str:
        settings = get_settings()
        return {
            "qwen_local": settings.qwen_tts_voice,
            "kokoro_local": settings.kokoro_tts_voice,
            "bark_local": settings.bark_tts_voice,
            "chatterbox_local": settings.chatterbox_tts_voice,
            "openai": settings.openai_tts_voice,
        }[self.resolve_provider(provider)]

    def model(self, provider: str | None = None) -> str:
        settings = get_settings()
        return {
            "qwen_local": settings.qwen_tts_model,
            "kokoro_local": settings.kokoro_tts_model,
            "bark_local": settings.bark_tts_model,
            "chatterbox_local": settings.chatterbox_tts_model,
            "openai": settings.openai_tts_model,
        }[self.resolve_provider(provider)]

    def _fetch_service_json(self, path: str, provider: str | None = None) -> dict | None:
        settings = get_settings()
        endpoint = {
            "qwen_local": settings.qwen_tts_endpoint,
            "bark_local": settings.bark_tts_endpoint,
            "chatterbox_local": settings.chatterbox_tts_endpoint,
        }.get(self.resolve_provider(provider))
        if not endpoint:
            return None
        url = endpoint.rsplit("/", 1)[0] + path
        try:
            with request.urlopen(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, json.JSONDecodeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def default_params(self, provider: str | None = None) -> dict | None:
        """Fetch the TTS service's env-configured tuning defaults, if reachable."""
        return self._fetch_service_json("/params", provider)

    def speaker_instructs(self, provider: str | None = None) -> dict | None:
        """Fetch each speaker's effective default instruct, if reachable."""
        return self._fetch_service_json("/speakers", provider)

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str | None = None,
        language: str | None = None,
        instruct: str | None = None,
        tts_params: dict | None = None,
        progress_callback: ProgressCallback | None = None,
        provider: str | None = None,
    ) -> Path:
        resolved_provider = self.resolve_provider(provider)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected_voice = voice or self.default_voice(resolved_provider)
        # A voice left over from the other provider (e.g. "Ryan" sent to OpenAI)
        # would fail synthesis, so fall back to this provider's default instead.
        if selected_voice not in {entry["id"] for entry in self.available_voices(resolved_provider)}:
            selected_voice = self.default_voice(resolved_provider)
        settings = get_settings()
        if resolved_provider == "qwen_local":
            return self._synthesize_local_service(
                "Qwen", settings.qwen_tts_endpoint, text, output_path, selected_voice, language, instruct, tts_params, progress_callback
            )
        if resolved_provider == "kokoro_local":
            return self._synthesize_local_service(
                "Kokoro", settings.kokoro_tts_endpoint, text, output_path, selected_voice, language, instruct, tts_params, progress_callback
            )
        if resolved_provider == "bark_local":
            return self._synthesize_local_service(
                "Bark", settings.bark_tts_endpoint, text, output_path, selected_voice, language, instruct, tts_params, progress_callback
            )
        if resolved_provider == "chatterbox_local":
            return self._synthesize_local_service(
                "Chatterbox", settings.chatterbox_tts_endpoint, text, output_path, selected_voice, language, instruct, tts_params, progress_callback
            )
        return self._synthesize_openai(text, output_path, selected_voice, instruct, progress_callback)

    def _synthesize_openai(
        self,
        text: str,
        output_path: Path,
        voice: str,
        instruct: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        # OpenAI has no chunking service, so the backend chunks the script itself
        # and measures each clip, giving subtitles the same timeline sidecar the
        # local services produce.
        chunks = parse_script(text)
        if not chunks:
            raise RuntimeError("Transcript contains no speakable content")
        timeline_path = timeline_sidecar_path(output_path)
        timeline_path.unlink(missing_ok=True)
        client = OpenAI(api_key=settings.openai_api_key)

        def synthesize_chunk(index_chunk: tuple[int, ScriptChunk]) -> tuple[int, Path]:
            index, chunk = index_chunk
            chunk_path = Path(scratch_dir) / f"chunk-{index:04d}.mp3"
            speech_args = {
                "model": settings.openai_tts_model,
                "voice": voice,
                "input": chunk.text,
                "response_format": "mp3",
            }
            if instruct:
                speech_args["instructions"] = instruct
            with client.audio.speech.with_streaming_response.create(**speech_args) as response:
                response.stream_to_file(chunk_path)
            return index, chunk_path

        with tempfile.TemporaryDirectory(prefix="openai-tts-") as scratch_dir:
            chunk_paths: dict[int, Path] = {}
            completed = 0
            with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
                for index, chunk_path in executor.map(synthesize_chunk, enumerate(chunks)):
                    chunk_paths[index] = chunk_path
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(chunks))
            durations = [self._probe_duration(chunk_paths[index]) for index in range(len(chunks))]
            self._assemble_openai_audio(chunks, chunk_paths, output_path)

        timeline: list[dict] = []
        cursor = 0.0
        for chunk, duration in zip(chunks, durations):
            timeline.append(
                {
                    "text": chunk.display_text,
                    "start": round(cursor, 3),
                    "end": round(cursor + duration, 3),
                }
            )
            cursor += duration + chunk.gap_after_ms / 1000
        timeline_path.write_text(json.dumps(timeline, ensure_ascii=False), encoding="utf-8")
        return output_path

    def _probe_duration(self, audio_path: Path) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "ffprobe failed")
        return float(result.stdout.strip())

    def _assemble_openai_audio(self, chunks: list[ScriptChunk], chunk_paths: dict[int, Path], output_path: Path) -> None:
        """Concatenate the per-chunk clips with their silence gaps into one mp3."""
        command = ["ffmpeg", "-y"]
        filter_inputs: list[str] = []
        filters: list[str] = []
        input_count = 0
        for index, chunk in enumerate(chunks):
            command += ["-i", str(chunk_paths[index])]
            # The concat filter needs uniform sample rates and layouts.
            filters.append(f"[{input_count}:a]aresample=24000,aformat=channel_layouts=mono[a{input_count}]")
            filter_inputs.append(f"[a{input_count}]")
            input_count += 1
            if chunk.gap_after_ms > 0:
                command += ["-f", "lavfi", "-t", f"{chunk.gap_after_ms / 1000:.3f}", "-i", "anullsrc=r=24000:cl=mono"]
                filter_inputs.append(f"[{input_count}:a]")
                input_count += 1
        filters.append(f"{''.join(filter_inputs)}concat=n={len(filter_inputs)}:v=0:a=1[out]")
        command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-b:a", "192k", str(output_path)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "FFmpeg concat failed")

    def _synthesize_local_service(
        self,
        service_name: str,
        endpoint: str | None,
        text: str,
        output_path: Path,
        voice: str,
        language: str | None = None,
        instruct: str | None = None,
        tts_params: dict | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        if not endpoint:
            raise RuntimeError(f"{service_name.upper()}_TTS_ENDPOINT is not configured")
        request_id = uuid.uuid4().hex
        request_payload = {
            "text": text,
            "input": text,
            "voice": voice,
            "response_format": "mp3",
            "request_id": request_id,
            "include_timeline": True,
        }
        if language:
            request_payload["language"] = language
        if instruct:
            request_payload["instruct"] = instruct
        if tts_params:
            request_payload["params"] = tts_params
        payload = json.dumps(request_payload).encode("utf-8")
        req = request.Request(
            endpoint,
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
            progress_url = endpoint.rsplit("/", 1)[0] + f"/progress/{request_id}"
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
            raise RuntimeError(f"{service_name} TTS endpoint is unavailable: {exc}") from exc
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
            raise RuntimeError("TTS service returned invalid JSON") from exc
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
                raise RuntimeError(f"Could not download TTS audio: {exc}") from exc
            return
        raise RuntimeError("TTS service JSON must include audio_base64, audio, audio_url, or url")
