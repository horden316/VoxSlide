"""Sentence chunker for TTS providers without a chunking service (OpenAI).

Keep the parsing rules in sync with parse_script in qwen_tts_service/server.py
and kokoro_tts_service/server.py. Unlike the local services, OpenAI clips are
not edge-trimmed, so the inserted gaps here are small paddings on top of each
clip's natural trailing silence.
"""

from dataclasses import dataclass
import re

from .pause_markers import PAUSE_DEFAULT_MS, PAUSE_MARKER
from .pronunciation_markers import PRONUNCIATION_MARKER

MAX_CHARS_PER_CHUNK = 200
MIN_CHUNK_CHARS = 80
SENTENCE_GAP_MS = 250
SEMICOLON_GAP_MS = 150
PARAGRAPH_GAP_MS = 700
WRAP_GAP_MS = 100

SENTENCE_SPLIT = re.compile(r"(?<=[。！？；])\s*|(?<=[!?;.])\s+")
CJK_CHAR = re.compile(r"[　-ヿ㐀-䶿一-鿿가-힯＀-￯]")

# Each pronunciation marker is swapped for one private-use character before
# chunking so sentence splitting and wrapping can never cut a marker apart.
PRONUNCIATION_PLACEHOLDER_BASE = 0xE000
PRONUNCIATION_PLACEHOLDER = re.compile("[-]")


@dataclass
class ScriptChunk:
    text: str
    display_text: str
    gap_after_ms: int
    explicit_gap: bool = False


def extract_pronunciation_markers(text: str) -> tuple[str, list[tuple[str, str]]]:
    pronunciations: list[tuple[str, str]] = []

    def replace(match: re.Match) -> str:
        display = match.group(1).strip()
        spoken = match.group(2).strip() or display
        pronunciations.append((display, spoken))
        return chr(PRONUNCIATION_PLACEHOLDER_BASE + len(pronunciations) - 1)

    return PRONUNCIATION_MARKER.sub(replace, text), pronunciations


def expand_pronunciations(text: str, pronunciations: list[tuple[str, str]], spoken: bool) -> str:
    def replace(match: re.Match) -> str:
        index = ord(match.group(0)) - PRONUNCIATION_PLACEHOLDER_BASE
        if 0 <= index < len(pronunciations):
            display, read = pronunciations[index]
            return read if spoken else display
        return ""

    return PRONUNCIATION_PLACEHOLDER.sub(replace, text)


def join_sentences(left: str, right: str) -> str:
    if not left:
        return right
    if CJK_CHAR.match(left[-1]) and CJK_CHAR.match(right[0]):
        return f"{left}{right}"
    return f"{left} {right}"


def boundary_gap_ms(sentence: str) -> int:
    if sentence and sentence[-1] in "；;":
        return SEMICOLON_GAP_MS
    return SENTENCE_GAP_MS


def parse_segment(segment: str) -> list[ScriptChunk]:
    """Split marker-free text into sentence chunks with punctuation-based gaps."""
    normalized = re.sub(r"\s+", " ", segment).strip()
    if not normalized:
        return []
    sentences = [part.strip() for part in SENTENCE_SPLIT.split(normalized) if part.strip()]
    merged: list[str] = []
    for sentence in sentences:
        if merged and len(merged[-1]) < MIN_CHUNK_CHARS and len(merged[-1]) + len(sentence) <= MAX_CHARS_PER_CHUNK:
            merged[-1] = join_sentences(merged[-1], sentence)
        else:
            merged.append(sentence)
    if len(merged) >= 2 and len(merged[-1]) < MIN_CHUNK_CHARS and len(merged[-2]) + len(merged[-1]) <= MAX_CHARS_PER_CHUNK:
        trailing = merged.pop()
        merged[-1] = join_sentences(merged[-1], trailing)
    chunks: list[ScriptChunk] = []
    for sentence in merged:
        parts = wrap_text(sentence, MAX_CHARS_PER_CHUNK) if len(sentence) > MAX_CHARS_PER_CHUNK else [sentence]
        for part in parts[:-1]:
            chunks.append(ScriptChunk(part, part, WRAP_GAP_MS))
        chunks.append(ScriptChunk(parts[-1], parts[-1], boundary_gap_ms(sentence)))
    return chunks


def parse_paragraph(paragraph: str) -> list[ScriptChunk]:
    chunks: list[ScriptChunk] = []
    position = 0
    for match in PAUSE_MARKER.finditer(paragraph):
        pause_ms = int(match.group(1)) if match.group(1) else PAUSE_DEFAULT_MS
        segment_chunks = parse_segment(paragraph[position : match.start()])
        if segment_chunks:
            segment_chunks[-1].gap_after_ms = pause_ms
            segment_chunks[-1].explicit_gap = True
            chunks.extend(segment_chunks)
        elif chunks:
            chunks[-1].gap_after_ms += pause_ms
            chunks[-1].explicit_gap = True
        position = match.end()
    chunks.extend(parse_segment(paragraph[position:]))
    return chunks


def parse_script(text: str) -> list[ScriptChunk]:
    protected, pronunciations = extract_pronunciation_markers(text)
    chunks: list[ScriptChunk] = []
    for paragraph in re.split(r"\n\s*\n+", protected.strip()):
        paragraph_chunks = parse_paragraph(paragraph)
        if not paragraph_chunks:
            continue
        if chunks and not chunks[-1].explicit_gap:
            chunks[-1].gap_after_ms = PARAGRAPH_GAP_MS
        chunks.extend(paragraph_chunks)
    if chunks and not chunks[-1].explicit_gap:
        chunks[-1].gap_after_ms = 0
    for chunk in chunks:
        placeholder_text = chunk.text
        chunk.text = expand_pronunciations(placeholder_text, pronunciations, spoken=True)
        chunk.display_text = expand_pronunciations(placeholder_text, pronunciations, spoken=False)
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
