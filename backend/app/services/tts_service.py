from pathlib import Path
from openai import OpenAI
from ..config import get_settings


class TtsService:
    def synthesize(self, text: str, output_path: Path) -> Path:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        client = OpenAI(api_key=settings.openai_api_key)
        with client.audio.speech.with_streaming_response.create(
            model=settings.openai_tts_model,
            voice=settings.openai_tts_voice,
            input=text,
            response_format="mp3",
        ) as response:
            response.stream_to_file(output_path)
        return output_path
