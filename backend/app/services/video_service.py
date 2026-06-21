from pathlib import Path
import subprocess


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
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
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
