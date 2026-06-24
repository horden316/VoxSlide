from pathlib import Path
import re
import subprocess

from ..config import get_settings


class VideoService:
    def _run(self, command: list[str]) -> None:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or "FFmpeg command failed")

    def probe_duration(self, audio_path: Path) -> float:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "ffprobe failed")
        return float(result.stdout.strip())

    def render_segment(self, image_path: Path, audio_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        settings = get_settings()
        video_options = self._video_encoder_options(settings.video_encoder)
        self._run(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                str(image_path),
                "-i",
                str(audio_path),
                *video_options,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-pix_fmt",
                "yuv420p",
                "-shortest",
                str(output_path),
            ]
        )
        return output_path

    def _video_encoder_options(self, encoder: str) -> list[str]:
        if encoder == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"]
        if encoder == "libx264":
            return ["-c:v", "libx264", "-tune", "stillimage"]
        return ["-c:v", encoder]

    def concat_segments(self, segments: list[Path], output_path: Path) -> Path:
        list_path = output_path.parent / "segments.txt"
        list_path.write_text(
            "\n".join(f"file '{segment.resolve().as_posix()}'" for segment in segments),
            encoding="utf-8",
        )
        self._run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_path),
            ]
        )
        return output_path

    def write_srt(self, captions: list[tuple[str, float]], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        position = 0.0
        entries: list[str] = []
        cue_index = 1
        for text, duration in captions:
            chunks = self._split_caption_text(text)
            if not chunks:
                position += max(duration, 0.0)
                continue
            chunk_weights = [max(len(chunk), 1) for chunk in chunks]
            total_weight = sum(chunk_weights)
            cue_start = position
            caption_end = position + max(duration, 0.0)
            for chunk_position, (chunk, weight) in enumerate(zip(chunks, chunk_weights), start=1):
                cue_duration = max(duration, 0.0) * weight / total_weight
                cue_end = cue_start + cue_duration
                if chunk_position == len(chunks):
                    cue_end = caption_end
                entries.append(
                    "\n".join(
                        [
                            str(cue_index),
                            f"{self._format_srt_timestamp(cue_start)} --> {self._format_srt_timestamp(cue_end)}",
                            chunk,
                        ]
                    )
                )
                cue_index += 1
                cue_start = cue_end
            position = caption_end
        output_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
        return output_path

    def _split_caption_text(self, text: str, max_chars: int = 48) -> list[str]:
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return []
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[。！？!?；;])\s*", normalized) if sentence.strip()]
        chunks: list[str] = []
        for sentence in sentences:
            chunks.extend(self._wrap_caption_sentence(sentence, max_chars))
        return chunks

    def _wrap_caption_sentence(self, sentence: str, max_chars: int) -> list[str]:
        if len(sentence) <= max_chars:
            return [sentence]
        chunks: list[str] = []
        remaining = sentence
        while len(remaining) > max_chars:
            split_at = self._caption_split_position(remaining, max_chars)
            chunks.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            chunks.append(remaining)
        return chunks

    def _caption_split_position(self, text: str, max_chars: int) -> int:
        window = text[: max_chars + 1]
        for pattern in [r"[,，、]\s*", r"\s+"]:
            matches = list(re.finditer(pattern, window))
            if matches:
                split_at = matches[-1].end()
                if split_at > 0:
                    return split_at
        return max_chars

    def _format_srt_timestamp(self, seconds: float) -> str:
        milliseconds = round(seconds * 1000)
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        whole_seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"
