import json
import re

# Keep in sync with PRONUNCIATION_MARKER in qwen_tts_service/server.py and
# kokoro_tts_service/server.py.
# [dis:display text|read:spoken text] — subtitles show the dis side, TTS reads
# the read side.
PRONUNCIATION_MARKER = re.compile(r"\[dis:([^|\[\]]*)\|read:([^\[\]]*)\]", re.IGNORECASE)


def to_read_text(text: str) -> str:
    """Resolve markers to the text the TTS should pronounce."""
    return PRONUNCIATION_MARKER.sub(lambda match: match.group(2).strip() or match.group(1).strip(), text)


def to_display_text(text: str) -> str:
    """Resolve markers to the text subtitles should show."""
    return PRONUNCIATION_MARKER.sub(lambda match: match.group(1).strip(), text)


def parse_glossary(raw: str | None) -> list[tuple[str, str]]:
    """Decode a project's stored glossary JSON into usable (display, read) pairs."""
    try:
        data = json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    entries: list[tuple[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        display = str(item.get("display") or "").strip()
        read = str(item.get("read") or "").strip()
        # Brackets or pipes inside a term would corrupt the marker it gets
        # wrapped into, so such entries are ignored.
        if display and read and display != read and not re.search(r"[\[\]|]", display + read):
            entries.append((display, read))
    return entries


def apply_glossary(text: str, entries: list[tuple[str, str]]) -> str:
    """Wrap plain occurrences of glossary terms in pronunciation markers.

    Text already inside a pronunciation marker is left untouched, so inline
    markers always win over the glossary.
    """
    if not text or not entries:
        return text
    # Longest display term first so overlapping terms prefer the longest match.
    ordered = sorted(entries, key=lambda entry: len(entry[0]), reverse=True)
    read_by_display = {display: read for display, read in reversed(ordered)}
    term_pattern = re.compile("|".join(re.escape(display) for display, _ in ordered))

    def wrap_terms(segment: str) -> str:
        return term_pattern.sub(
            lambda match: f"[dis:{match.group(0)}|read:{read_by_display[match.group(0)]}]",
            segment,
        )

    result: list[str] = []
    position = 0
    for match in PRONUNCIATION_MARKER.finditer(text):
        result.append(wrap_terms(text[position : match.start()]))
        result.append(match.group(0))
        position = match.end()
    result.append(wrap_terms(text[position:]))
    return "".join(result)
