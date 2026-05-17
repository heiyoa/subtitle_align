from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .align import Word
from .subtitles import Cue


DEFAULT_HALLUCINATION_LEXICON = {
    "working",
    "okay",
    "ok",
    "yeah",
    "oh",
    "uh",
    "um",
    "hmm",
    "hey",
    "right",
    "sorry",
}


@dataclass(frozen=True)
class RemovedWordSample:
    word: str
    start: float | None
    end: float | None
    reason: str

    def to_jsonable(self) -> dict:
        return {"word": self.word, "start": self.start, "end": self.end, "reason": self.reason}


def load_hallucination_lexicon(path: str | None) -> set[str]:
    if not path:
        return set(DEFAULT_HALLUCINATION_LEXICON)
    p = Path(path)
    if not p.exists():
        return set(DEFAULT_HALLUCINATION_LEXICON)
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return set(DEFAULT_HALLUCINATION_LEXICON)
    if p.suffix.lower() == ".json":
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                extra = {str(x).strip().lower() for x in obj if str(x).strip()}
                return set(DEFAULT_HALLUCINATION_LEXICON) | extra
        except Exception:
            return set(DEFAULT_HALLUCINATION_LEXICON)
    extra = {line.strip().lower() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")}
    return set(DEFAULT_HALLUCINATION_LEXICON) | extra


_WORD_CLEAN = re.compile(r"[^a-zA-Z']+")


def normalize_word_for_lexicon(word: str) -> str:
    w = (word or "").strip().lower()
    w = _WORD_CLEAN.sub("", w)
    return w


def apply_word_duration_gate(
    words: list[Word],
    *,
    max_word_dur: float,
    removed_word_placeholder: str,
    hallucination_lexicon: set[str],
) -> tuple[list[Word], list[RemovedWordSample]]:
    """
    Removes suspect long-duration words (likely non-speech / hallucination).

    If removed_word_placeholder == "ellipsis", inserts a short " …" token at the original start time.
    """
    kept: list[Word] = []
    removed: list[RemovedWordSample] = []
    for w in words:
        if w.start is None or w.end is None:
            kept.append(w)
            continue
        dur = float(w.end) - float(w.start)
        if dur <= max_word_dur:
            kept.append(w)
            continue

        norm = normalize_word_for_lexicon(w.word)
        reason = "word_duration_gate"
        if norm in hallucination_lexicon:
            reason = "word_duration_gate+lexicon"
        removed.append(RemovedWordSample(word=w.word, start=float(w.start), end=float(w.end), reason=reason))
        if removed_word_placeholder == "ellipsis":
            s = float(w.start)
            e = min(float(w.end), s + 0.05)
            kept.append(Word(word=" …", start=s, end=e, probability=None))
    kept.sort(key=lambda x: (x.start is None, x.start if x.start is not None else 0.0))
    return kept, removed


def apply_island_words_gate(
    words: list[Word],
    *,
    hole_gap_sec: float,
    island_window_sec: float,
    island_max_words: int,
    removed_word_placeholder: str,
    hallucination_lexicon: set[str],
) -> tuple[list[Word], list[RemovedWordSample], list[str]]:
    """
    Removes "island words": tiny clusters (1-2 words by default) surrounded by large gaps.
    """
    usable = [w for w in words if w.start is not None and w.end is not None]
    usable.sort(key=lambda w: float(w.start))
    if not usable:
        return words, [], []

    clusters: list[list[Word]] = []
    cur: list[Word] = [usable[0]]
    prev_end = float(usable[0].end)
    for w in usable[1:]:
        gap = float(w.start) - prev_end
        if gap >= hole_gap_sec:
            clusters.append(cur)
            cur = [w]
        else:
            cur.append(w)
        prev_end = max(prev_end, float(w.end))
    if cur:
        clusters.append(cur)

    remove_ranges: list[tuple[float, float, list[Word]]] = []
    warnings: list[str] = []
    for c in clusters:
        c_start = float(c[0].start)
        c_end = float(c[-1].end)
        c_dur = c_end - c_start
        if c_dur > island_window_sec or len(c) > island_max_words:
            continue

        left_gap = c_start - _prev_end_before(usable, c_start)
        right_gap = _next_start_after(usable, c_end) - c_end
        if left_gap >= hole_gap_sec and right_gap >= hole_gap_sec:
            remove_ranges.append((c_start, c_end, c))
            sample_words = ", ".join(w.word.strip() for w in c)[:80]
            warnings.append(f"gate_island_words_removed [{c_start:.2f},{c_end:.2f}] words={sample_words}")

    if not remove_ranges:
        return words, [], []

    removed: list[RemovedWordSample] = []
    to_remove: set[tuple[float, float]] = set()
    for s, e, c in remove_ranges:
        to_remove.add((s, e))
        lex_hit = any(normalize_word_for_lexicon(w.word) in hallucination_lexicon for w in c)
        for w in c:
            removed.append(
                RemovedWordSample(
                    word=w.word,
                    start=float(w.start) if w.start is not None else None,
                    end=float(w.end) if w.end is not None else None,
                    reason=("island_words_gate+lexicon" if lex_hit else "island_words_gate"),
                )
            )

    kept: list[Word] = []
    for w in words:
        if w.start is None or w.end is None:
            kept.append(w)
            continue
        inside = any(s <= float(w.start) < e for (s, e) in to_remove)
        if not inside:
            kept.append(w)
            continue
        # removed: optionally insert ellipsis at range start; do this once per range

    if removed_word_placeholder == "ellipsis":
        # insert one ellipsis per removed range
        for s, e in sorted(to_remove):
            kept.append(Word(word=" …", start=s, end=min(e, s + 0.05), probability=None))

    kept.sort(key=lambda x: (x.start is None, x.start if x.start is not None else 0.0))
    return kept, removed, warnings


def enforce_caption_duration_hard(
    cues: list[Cue],
    *,
    max_caption_dur_hard: float,
    gap_threshold: float,
) -> tuple[list[Cue], int, list[str]]:
    """
    Hard cap: if any cue duration > max_caption_dur_hard, it must be resegmented.
    Prefer splitting on large word gaps; otherwise split at nearest word boundary.
    If still impossible, drop the cue and emit warning.
    """
    out: list[Cue] = []
    num_resegmented = 0
    warnings: list[str] = []

    for c in cues:
        if c.duration <= max_caption_dur_hard:
            out.append(c)
            continue

        if not c.words:
            warnings.append(f"gate_caption_duration_drop_no_words start={c.start:.2f} end={c.end:.2f} dur={c.duration:.2f}")
            num_resegmented += 1
            continue

        pieces = _split_cue_by_words(c, max_caption_dur_hard=max_caption_dur_hard, gap_threshold=gap_threshold)
        if not pieces:
            warnings.append(f"gate_caption_duration_drop start={c.start:.2f} end={c.end:.2f} dur={c.duration:.2f}")
            num_resegmented += 1
            continue

        out.extend(pieces)
        num_resegmented += 1

    return out, num_resegmented, warnings


def _split_cue_by_words(cue: Cue, *, max_caption_dur_hard: float, gap_threshold: float) -> list[Cue]:
    # Greedy chunking with "best split" at last big gap within the chunk.
    words = [w for w in cue.words if w.start is not None and w.end is not None]
    if not words:
        return []

    pieces: list[Cue] = []
    i = 0
    while i < len(words):
        start = float(words[i].start)
        j = i
        last_ok = i
        last_split = None
        best_gap = 0.0
        prev_end = float(words[i].end)
        while j + 1 < len(words):
            nxt = words[j + 1]
            proposed_end = float(nxt.end)
            if (proposed_end - start) > max_caption_dur_hard:
                break
            gap = float(nxt.start) - prev_end
            if gap >= gap_threshold and gap >= best_gap:
                best_gap = gap
                last_split = j + 1
            prev_end = max(prev_end, float(nxt.end))
            j += 1
            last_ok = j

        if last_ok == i and (float(words[i].end) - start) > max_caption_dur_hard:
            # single word too long; cannot split
            return []

        split_at = last_split if last_split is not None else (last_ok + 1)
        chunk = words[i:split_at]
        text = "".join(w.text for w in chunk).strip()
        if text:
            pieces.append(Cue(start=float(chunk[0].start), end=float(chunk[-1].end), text=text, words=list(chunk)))
        i = split_at

    return pieces


def _prev_end_before(words: list[Word], t: float) -> float:
    prev = 0.0
    for w in words:
        if w.end is None or w.start is None:
            continue
        if float(w.end) <= t:
            prev = max(prev, float(w.end))
    return prev


def _next_start_after(words: list[Word], t: float) -> float:
    nxt = float("inf")
    for w in words:
        if w.start is None:
            continue
        if float(w.start) >= t:
            nxt = min(nxt, float(w.start))
    return nxt if nxt != float("inf") else t

