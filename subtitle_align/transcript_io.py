from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .align import Segment, Word, segments_from_asr_jsonable


@dataclass(frozen=True)
class LoadedTranscript:
    raw: dict[str, Any]
    segments: list[dict[str, Any]]
    segments_obj: list[Segment]
    words: list[Word]
    language: str | None


def load_transcript(path: Path) -> LoadedTranscript:
    """
    Loads transcript JSON from either:
    - old format: {"segments": [...], "language": "..."} or {"segments_raw": [...], ...}
    - new format: {"segments_primary": [...], "segments": [...], ...}
    """
    obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    segments = _pick_segments(obj)
    seg_obj = segments_from_asr_jsonable(segments)
    words: list[Word] = []
    for s in seg_obj:
        for w in s.words:
            words.append(w)
    words.sort(key=lambda x: (x.start is None, x.start if x.start is not None else 0.0))
    lang = None
    if isinstance(obj.get("language"), str):
        lang = obj.get("language")
    elif isinstance(obj.get("asr"), dict) and isinstance(obj["asr"].get("detected_language"), str):
        lang = obj["asr"].get("detected_language")
    return LoadedTranscript(raw=obj, segments=segments, segments_obj=seg_obj, words=words, language=lang)


def _pick_segments(obj: dict[str, Any]) -> list[dict[str, Any]]:
    for k in ("segments", "segments_primary", "segments_raw"):
        v = obj.get(k)
        if isinstance(v, list) and v:
            if isinstance(v[0], dict):
                return v  # type: ignore[return-value]
    # fallback: empty list
    return []

