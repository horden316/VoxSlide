from pathlib import Path
import re
import subprocess

from ..config import get_settings
from .pause_markers import PAUSE_DEFAULT_MS, PAUSE_MARKER


class VideoService:
    # Mirrors the qwen service paragraph gap so cue timing tracks the audio.
    PARAGRAPH_PAUSE_MS = 1000
    # Approximate speech rate used to convert pause seconds into caption weight units.
    PAUSE_WEIGHT_CHARS_PER_SECOND = 4.0

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

    def write_srt(self, captions: list[tuple[str, float, float]], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        position = 0.0
        entries: list[str] = []
        cue_index = 1
        for text, audio_duration, slot_duration in captions:
            audio_duration = max(audio_duration, 0.0)
            # The rendered segment can differ slightly from its audio (frame
            # rounding, encoder padding); advancing by the real slot keeps
            # later pages aligned with the concatenated video.
            slot_duration = slot_duration if slot_duration > 0 else audio_duration
            caption_span = min(audio_duration, slot_duration)
            chunks = self._split_caption_text(text)
            if not chunks:
                position += slot_duration
                continue
            chunk_weights = [
                self._caption_weight(chunk) + pause_seconds * self.PAUSE_WEIGHT_CHARS_PER_SECOND
                for chunk, pause_seconds in chunks
            ]
            total_weight = sum(chunk_weights)
            cue_start = position
            caption_end = position + caption_span
            for chunk_position, ((chunk, _), weight) in enumerate(zip(chunks, chunk_weights), start=1):
                cue_duration = caption_span * weight / total_weight
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
            position += slot_duration
        output_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
        return output_path

    def _split_caption_text(self, text: str, max_chars: int = 48) -> list[tuple[str, float]]:
        """Split into caption chunks, each paired with the pause seconds that follow it."""
        annotated = re.sub(r"\n\s*\n+", f" [pause:{self.PARAGRAPH_PAUSE_MS}] ", text.strip())
        chunks: list[tuple[str, float]] = []
        position = 0
        for match in PAUSE_MARKER.finditer(annotated):
            pause_ms = int(match.group(1)) if match.group(1) else PAUSE_DEFAULT_MS
            self._append_caption_segment(chunks, annotated[position : match.start()], max_chars, pause_ms / 1000)
            position = match.end()
        self._append_caption_segment(chunks, annotated[position:], max_chars, 0.0)
        return chunks

    def _append_caption_segment(
        self,
        chunks: list[tuple[str, float]],
        segment: str,
        max_chars: int,
        pause_after: float,
    ) -> None:
        segment_chunks = self._chunk_caption_segment(segment, max_chars)
        for chunk in segment_chunks[:-1]:
            chunks.append((chunk, 0.0))
        if segment_chunks:
            chunks.append((segment_chunks[-1], pause_after))
        elif chunks:
            last_text, last_pause = chunks[-1]
            chunks[-1] = (last_text, last_pause + pause_after)

    def _chunk_caption_segment(self, text: str, max_chars: int) -> list[str]:
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return []
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？；])\s*|(?<=[.!?;])\s+|(?<=[.!?][\"'”’])\s+", normalized)
            if sentence.strip()
        ]
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._wrap_caption_sentence(sentence, max_chars))
                continue
            candidate = self._join_caption_parts(current, sentence)
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _join_caption_parts(self, left: str, right: str) -> str:
        if not left:
            return right
        cjk = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")
        if cjk.match(left[-1]) and cjk.match(right[0]):
            return f"{left}{right}"
        return f"{left} {right}"

    def _caption_weight(self, chunk: str) -> float:
        cjk_chars = len(re.findall(r"[぀-ヿ㐀-䶿一-鿿가-힯]", chunk))
        latin_words = re.findall(r"[A-Za-z0-9']+", chunk)
        latin_weight = sum(max(1.0, len(word) / 3) for word in latin_words)
        # Sentence enders now carry a real ~550ms inserted gap in the TTS audio.
        sentence_pauses = len(re.findall(r"[。！？!?；;.]", chunk))
        clause_pauses = len(re.findall(r"[,，、:：]", chunk))
        return max(cjk_chars + latin_weight + sentence_pauses * 2.4 + clause_pauses * 0.8, 1.0)

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
        if len(chunks) > 1 and len(chunks[-1]) < 16 and len(chunks[-2]) + len(chunks[-1]) + 1 <= 64:
            chunks[-2] = f"{chunks[-2]} {chunks[-1]}"
            chunks.pop()
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
