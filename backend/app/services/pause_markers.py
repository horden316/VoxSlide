import re

# Keep in sync with PAUSE_MARKER in qwen_tts_service/server.py.
PAUSE_MARKER = re.compile(r"\[pause(?::\s*(\d+)\s*(?:ms)?)?\]", re.IGNORECASE)
PAUSE_DEFAULT_MS = 1000


def strip_pause_markers(text: str) -> str:
    stripped = PAUSE_MARKER.sub(" ", text)
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()
