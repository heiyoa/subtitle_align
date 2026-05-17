from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Word:
    word: str
    start: Optional[float]
    end: Optional[float]
    probability: Optional[float] = None


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]


@dataclass(frozen=True)
class Hole:
    core_start: float
    core_end: float
    run_start: float
    run_end: float


def segments_from_asr_jsonable(segments: list[dict[str, Any]]) -> list[Segment]:
    out: list[Segment] = []
    for s in segments:
        words: list[Word] = []
        for w in (s.get("words") or []):
            words.append(
                Word(
                    word=str(w.get("word", "")),
                    start=w.get("start", None),
                    end=w.get("end", None),
                    probability=w.get("probability", None),
                )
            )
        out.append(Segment(start=float(s.get("start", 0.0)), end=float(s.get("end", 0.0)), text=str(s.get("text", "")), words=words))
    return out


def collect_words(segments: list[Segment], *, offset_sec: float = 0.0) -> list[Word]:
    words: list[Word] = []
    for s in segments:
        for w in s.words:
            if w.start is None or w.end is None:
                words.append(w)
                continue
            words.append(
                Word(
                    word=w.word,
                    start=float(w.start) + offset_sec,
                    end=float(w.end) + offset_sec,
                    probability=w.probability,
                )
            )
    words.sort(key=lambda x: (x.start is None, x.start if x.start is not None else 0.0))
    return words


def word_timestamp_coverage(words: list[Word]) -> float:
    if not words:
        return 0.0
    ok = sum(1 for w in words if w.start is not None and w.end is not None)
    return ok / len(words)


def detect_holes_from_words(
    words: list[Word],
    *,
    duration_sec: float,
    hole_gap_sec: float,
    pad_sec: float = 0.2,
) -> list[Hole]:
    usable = [w for w in words if w.start is not None and w.end is not None]
    usable.sort(key=lambda w: float(w.start))
    if duration_sec <= 0:
        return []
    if not usable:
        return [_make_hole(0.0, duration_sec, duration_sec=duration_sec, pad_sec=pad_sec)]

    holes: list[Hole] = []
    prev_end = 0.0
    for w in usable:
        start = float(w.start)
        if (start - prev_end) >= hole_gap_sec:
            holes.append(_make_hole(prev_end, start, duration_sec=duration_sec, pad_sec=pad_sec))
        prev_end = max(prev_end, float(w.end))

    if (duration_sec - prev_end) >= hole_gap_sec:
        holes.append(_make_hole(prev_end, duration_sec, duration_sec=duration_sec, pad_sec=pad_sec))

    return _merge_holes(holes)


def detect_holes_from_segments(
    segments: list[Segment],
    *,
    duration_sec: float,
    hole_gap_sec: float,
    pad_sec: float = 0.2,
) -> list[Hole]:
    segs = sorted(segments, key=lambda s: s.start)
    if duration_sec <= 0:
        return []
    if not segs:
        return [_make_hole(0.0, duration_sec, duration_sec=duration_sec, pad_sec=pad_sec)]

    holes: list[Hole] = []
    prev_end = 0.0
    for s in segs:
        if (s.start - prev_end) >= hole_gap_sec:
            holes.append(_make_hole(prev_end, s.start, duration_sec=duration_sec, pad_sec=pad_sec))
        prev_end = max(prev_end, s.end)
    if (duration_sec - prev_end) >= hole_gap_sec:
        holes.append(_make_hole(prev_end, duration_sec, duration_sec=duration_sec, pad_sec=pad_sec))
    return _merge_holes(holes)


def merge_words_replace_interval(base: list[Word], patch: list[Word], *, start_sec: float, end_sec: float) -> list[Word]:
    """
    Replace base words whose start is in [start_sec, end_sec) with patch words in the same interval.
    """
    kept = [w for w in base if w.start is None or not (start_sec <= float(w.start) < end_sec)]
    added = [w for w in patch if w.start is not None and (start_sec <= float(w.start) < end_sec)]
    merged = kept + added
    merged.sort(key=lambda x: (x.start is None, x.start if x.start is not None else 0.0))
    return merged


def build_segments_from_words(words: list[Word], *, seg_gap_sec: float = 1.0, max_seg_sec: float = 15.0) -> list[Segment]:
    usable = [w for w in words if w.start is not None and w.end is not None]
    usable.sort(key=lambda w: float(w.start))
    if not usable:
        return []
    segments: list[Segment] = []
    cur: list[Word] = [usable[0]]
    seg_start = float(usable[0].start)
    prev_end = float(usable[0].end)
    for w in usable[1:]:
        gap = float(w.start) - prev_end
        if gap >= seg_gap_sec or (float(w.end) - seg_start) >= max_seg_sec:
            segments.append(_segment_from_words(cur))
            cur = [w]
            seg_start = float(w.start)
        else:
            cur.append(w)
        prev_end = max(prev_end, float(w.end))
    if cur:
        segments.append(_segment_from_words(cur))
    return segments


def segments_to_jsonable(segments: list[Segment]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, s in enumerate(segments):
        out.append(
            {
                "id": idx,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text,
                "words": [
                    {
                        "word": w.word,
                        "start": (round(float(w.start), 3) if w.start is not None else None),
                        "end": (round(float(w.end), 3) if w.end is not None else None),
                        "probability": w.probability,
                    }
                    for w in s.words
                ],
            }
        )
    return out


def _segment_from_words(words: list[Word]) -> Segment:
    start = float(words[0].start) if words[0].start is not None else 0.0
    end = float(words[-1].end) if words[-1].end is not None else start
    text = "".join(w.word for w in words).strip()
    return Segment(start=start, end=end, text=text, words=words)


def _make_hole(core_start: float, core_end: float, *, duration_sec: float, pad_sec: float) -> Hole:
    run_start = max(0.0, core_start - pad_sec)
    run_end = min(duration_sec, core_end + pad_sec)
    return Hole(core_start=core_start, core_end=core_end, run_start=run_start, run_end=run_end)


def _merge_holes(holes: list[Hole]) -> list[Hole]:
    if not holes:
        return holes
    hs = sorted(holes, key=lambda h: h.core_start)
    merged: list[Hole] = [hs[0]]
    for h in hs[1:]:
        prev = merged[-1]
        if h.core_start <= prev.core_end + 0.05:
            core_start = prev.core_start
            core_end = max(prev.core_end, h.core_end)
            run_start = min(prev.run_start, h.run_start)
            run_end = max(prev.run_end, h.run_end)
            merged[-1] = Hole(core_start=core_start, core_end=core_end, run_start=run_start, run_end=run_end)
        else:
            merged.append(h)
    return merged

