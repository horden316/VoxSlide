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
    # Cue sizing follows common subtitle conventions: 42 width units per line
    # (Latin chars count 1, CJK chars 2), at most two lines per cue.
    MAX_LINE_WIDTH = 42
    MAX_CUE_WIDTH = 84
    CJK_WIDE = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")

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
        # Stills carry no motion, so a low framerate cuts encode time ~3x with
        # no visible difference. -shortest overshoots by seconds at low fps,
        # so the segment is cut to the probed audio length instead.
        audio_duration = self.probe_duration(audio_path)
        self._run(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-framerate",
                str(settings.video_fps),
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
                "-t",
                f"{audio_duration:.3f}",
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

    def write_srt(
        self,
        captions: list[tuple[str, float, float, list[dict] | None]],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        position = 0.0
        entries: list[str] = []
        cue_index = 1
        for text, audio_duration, slot_duration, timeline in captions:
            audio_duration = max(audio_duration, 0.0)
            # The rendered segment can differ slightly from its audio (frame
            # rounding, encoder padding); advancing by the real slot keeps
            # later pages aligned with the concatenated video.
            slot_duration = slot_duration if slot_duration > 0 else audio_duration
            caption_span = min(audio_duration, slot_duration)
            cues = (
                self._cues_from_timeline(timeline, caption_span)
                if timeline
                else self._cues_from_weights(text, caption_span)
            )
            for cue_text, cue_start, cue_end in cues:
                entries.append(
                    "\n".join(
                        [
                            str(cue_index),
                            f"{self._format_srt_timestamp(position + cue_start)} --> {self._format_srt_timestamp(position + cue_end)}",
                            cue_text,
                        ]
                    )
                )
                cue_index += 1
            position += slot_duration
        output_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
        return output_path

    def _cues_from_timeline(self, timeline: list[dict], caption_span: float) -> list[tuple[str, float, float]]:
        """Anchor cues to the synthesized chunks' measured start/end times."""
        chunks: list[tuple[str, float, float]] = []
        for entry in timeline:
            chunk_text = str(entry.get("text") or "").strip()
            try:
                start = float(entry["start"])
                end = float(entry["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if not chunk_text or end <= start or start >= caption_span:
                continue
            chunks.append((chunk_text, start, min(end, caption_span)))
        cues: list[tuple[str, float, float]] = []
        for index, (chunk_text, start, end) in enumerate(chunks):
            # Keep the cue visible through the silence gap after its chunk, the
            # same way weight-based cues covered pauses.
            display_end = chunks[index + 1][1] if index + 1 < len(chunks) else caption_span
            display_end = max(display_end, end)
            cue_texts = self._chunk_caption_segment(chunk_text, self.MAX_CUE_WIDTH)
            if not cue_texts:
                continue
            # Within one chunk there is continuous speech, so character-weight
            # interpolation is a close approximation between the hard anchors.
            weights = [self._caption_weight(cue_text) for cue_text in cue_texts]
            total_weight = sum(weights)
            speech_span = max(end - start, 0.0)
            cue_start = start
            for cue_position, (cue_text, weight) in enumerate(zip(cue_texts, weights), start=1):
                cue_end = (
                    display_end
                    if cue_position == len(cue_texts)
                    else cue_start + speech_span * weight / total_weight
                )
                cues.append((self._wrap_cue_lines(cue_text), cue_start, cue_end))
                cue_start = cue_end
        return cues

    def _cues_from_weights(self, text: str, caption_span: float) -> list[tuple[str, float, float]]:
        """Fallback for audio without timing metadata: distribute by character weight."""
        chunks = self._split_caption_text(text)
        if not chunks:
            return []
        chunk_weights = [
            self._caption_weight(chunk) + pause_seconds * self.PAUSE_WEIGHT_CHARS_PER_SECOND
            for chunk, pause_seconds in chunks
        ]
        total_weight = sum(chunk_weights)
        cues: list[tuple[str, float, float]] = []
        cue_start = 0.0
        for chunk_position, ((chunk, _), weight) in enumerate(zip(chunks, chunk_weights), start=1):
            cue_end = (
                caption_span
                if chunk_position == len(chunks)
                else cue_start + caption_span * weight / total_weight
            )
            cues.append((self._wrap_cue_lines(chunk), cue_start, cue_end))
            cue_start = cue_end
        return cues

    def _split_caption_text(self, text: str) -> list[tuple[str, float]]:
        """Split into caption chunks, each paired with the pause seconds that follow it."""
        annotated = re.sub(r"\n\s*\n+", f" [pause:{self.PARAGRAPH_PAUSE_MS}] ", text.strip())
        chunks: list[tuple[str, float]] = []
        position = 0
        for match in PAUSE_MARKER.finditer(annotated):
            pause_ms = int(match.group(1)) if match.group(1) else PAUSE_DEFAULT_MS
            self._append_caption_segment(chunks, annotated[position : match.start()], pause_ms / 1000)
            position = match.end()
        self._append_caption_segment(chunks, annotated[position:], 0.0)
        return chunks

    def _append_caption_segment(
        self,
        chunks: list[tuple[str, float]],
        segment: str,
        pause_after: float,
    ) -> None:
        segment_chunks = self._chunk_caption_segment(segment, self.MAX_CUE_WIDTH)
        for chunk in segment_chunks[:-1]:
            chunks.append((chunk, 0.0))
        if segment_chunks:
            chunks.append((segment_chunks[-1], pause_after))
        elif chunks:
            last_text, last_pause = chunks[-1]
            chunks[-1] = (last_text, last_pause + pause_after)

    def _chunk_caption_segment(self, text: str, max_width: int) -> list[str]:
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
            if self._display_width(sentence) > max_width:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._wrap_caption_sentence(sentence, max_width))
                continue
            candidate = self._join_caption_parts(current, sentence)
            if current and self._display_width(candidate) > max_width:
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

    def _wrap_caption_sentence(self, sentence: str, max_width: int) -> list[str]:
        if self._display_width(sentence) <= max_width:
            return [sentence]
        chunks: list[str] = []
        remaining = sentence
        while self._display_width(remaining) > max_width:
            split_at = self._caption_split_position(remaining, max_width)
            chunks.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            chunks.append(remaining)
        if (
            len(chunks) > 1
            and self._display_width(chunks[-1]) < 16
            and self._display_width(chunks[-2]) + self._display_width(chunks[-1]) + 1 <= max_width
        ):
            trailing = chunks.pop()
            chunks[-1] = self._join_caption_parts(chunks[-1], trailing)
        return chunks

    def _caption_split_position(self, text: str, max_width: int) -> int:
        limit = self._width_prefix_length(text, max_width)
        window = text[: limit + 1]
        for pattern in [r"[,，、]\s*", r"\s+"]:
            matches = list(re.finditer(pattern, window))
            if matches:
                split_at = matches[-1].end()
                if split_at > 0:
                    return split_at
        return max(limit, 1)

    def _wrap_cue_lines(self, text: str) -> str:
        """Break a cue that exceeds one line into two balanced lines."""
        if self._display_width(text) <= self.MAX_LINE_WIDTH:
            return text
        best_split = None
        best_balance = None
        for match in re.finditer(r"[,，、:：]\s*|\s+", text):
            split_at = match.end()
            left = text[:split_at].strip()
            right = text[split_at:].strip()
            if not left or not right:
                continue
            left_width = self._display_width(left)
            right_width = self._display_width(right)
            if left_width > self.MAX_LINE_WIDTH or right_width > self.MAX_LINE_WIDTH:
                continue
            balance = abs(left_width - right_width)
            if best_balance is None or balance < best_balance:
                best_balance = balance
                best_split = split_at
        split_at = best_split if best_split is not None else self._caption_split_position(text, self.MAX_LINE_WIDTH)
        first = text[:split_at].strip()
        second = text[split_at:].strip()
        if not first or not second:
            return text
        return f"{first}\n{second}"

    def _display_width(self, text: str) -> int:
        return sum(2 if self.CJK_WIDE.match(char) else 1 for char in text)

    def _width_prefix_length(self, text: str, max_width: int) -> int:
        """Longest prefix length (in characters) whose display width fits max_width."""
        width = 0
        for index, char in enumerate(text):
            width += 2 if self.CJK_WIDE.match(char) else 1
            if width > max_width:
                return index
        return len(text)

    def _format_srt_timestamp(self, seconds: float) -> str:
        milliseconds = round(seconds * 1000)
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        whole_seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"
