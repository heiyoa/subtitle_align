from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class WordTS:
    text: str
    start: Optional[float]
    end: Optional[float]


@dataclass
class Cue:
    start: float
    end: float
    text: str
    words: list[WordTS]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


_PUNCT_STRONG = re.compile(r"[。！？.!?…]$")
_PUNCT_WEAK = re.compile(r"[,;:，；：]$")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?，。；：！？])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

_CONJ = {
    "and",
    "but",
    "or",
    "so",
    "because",
    "then",
    "however",
    "therefore",
    "also",
    "while",
    "although",
}


def build_cues_from_words(
    words: list[WordTS],
    *,
    gap_threshold: float,
    max_line_chars: int,
    min_duration: float,
    max_duration: float,
) -> list[Cue]:
    usable = [w for w in words if w.start is not None and w.end is not None]
    if not usable:
        return []

    cues: list[Cue] = []
    i = 0
    while i < len(usable):
        start = float(usable[i].start)
        cur_words: list[WordTS] = [usable[i]]
        i += 1

        while i < len(usable):
            prev = cur_words[-1]
            nxt = usable[i]
            gap = float(nxt.start) - float(prev.end)
            proposed_duration = float(nxt.end) - start

            if proposed_duration > max_duration:
                break

            last_token = _last_token_text(cur_words)

            if gap > gap_threshold and (float(prev.end) - start) >= min_duration:
                break

            if _PUNCT_STRONG.search(last_token) and (float(prev.end) - start) >= (min_duration * 0.75):
                break

            if _approx_chars(cur_words) >= int(max_line_chars * 1.8):
                if _PUNCT_WEAK.search(last_token) or _ends_with_conj(last_token):
                    if (float(prev.end) - start) >= (min_duration * 0.6):
                        break

            cur_words.append(nxt)
            i += 1

        end = float(cur_words[-1].end)
        text = _assemble_words_text(cur_words)
        text = _wrap_two_lines(text, max_line_chars=max_line_chars)
        cues.append(Cue(start=start, end=end, text=text, words=cur_words))

    cues = _merge_short_cues(cues, min_duration=min_duration, max_duration=max_duration, max_line_chars=max_line_chars)
    return cues


def build_cues_from_segments(
    segments: Iterable[dict],
    *,
    max_line_chars: int,
) -> list[Cue]:
    cues: list[Cue] = []
    for s in segments:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        text = _wrap_two_lines(text, max_line_chars=max_line_chars)
        cues.append(Cue(start=start, end=end, text=text, words=[]))
    return cues


def format_srt(cues: list[Cue]) -> str:
    lines: list[str] = []
    for idx, c in enumerate(cues, start=1):
        lines.append(str(idx))
        lines.append(f"{_ts_srt(c.start)} --> {_ts_srt(c.end)}")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_vtt(cues: list[Cue]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for c in cues:
        lines.append(f"{_ts_vtt(c.start)} --> {_ts_vtt(c.end)}")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _ts_srt(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    ms = int(round(sec * 1000.0))
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    ms = int(round(sec * 1000.0))
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _assemble_words_text(words: list[WordTS]) -> str:
    text = "".join(w.text for w in words).strip()
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text


def _wrap_two_lines(text: str, *, max_line_chars: int) -> str:
    text = _MULTI_SPACE.sub(" ", text.strip())
    if len(text) <= max_line_chars:
        return text
    words = [w for w in text.split(" ") if w]
    if len(words) <= 1:
        return text

    best = None
    best_score = None
    for i in range(1, len(words)):
        l1 = " ".join(words[:i])
        l2 = " ".join(words[i:])
        overflow = max(0, len(l1) - max_line_chars) + max(0, len(l2) - max_line_chars)
        score = (overflow * 10) + max(len(l1), len(l2))
        if best_score is None or score < best_score:
            best_score = score
            best = (l1, l2)
    if best is None:
        return text
    l1, l2 = best
    return f"{l1}\n{l2}".strip()


def _approx_chars(words: list[WordTS]) -> int:
    return len(_assemble_words_text(words))


def _last_token_text(words: list[WordTS]) -> str:
    if not words:
        return ""
    return words[-1].text.strip()


def _ends_with_conj(last_token: str) -> bool:
    t = re.sub(r"[^A-Za-z]+", "", last_token).lower()
    return t in _CONJ


def _merge_short_cues(
    cues: list[Cue],
    *,
    min_duration: float,
    max_duration: float,
    max_line_chars: int,
) -> list[Cue]:
    if not cues:
        return cues
    merged: list[Cue] = []
    i = 0
    while i < len(cues):
        cur = cues[i]
        if cur.duration >= min_duration or i == len(cues) - 1:
            merged.append(cur)
            i += 1
            continue

        nxt = cues[i + 1]
        combined_duration = nxt.end - cur.start
        if combined_duration <= max_duration:
            combined_words = cur.words + nxt.words
            combined_text = (cur.text.replace("\n", " ") + " " + nxt.text.replace("\n", " ")).strip()
            combined_text = _wrap_two_lines(combined_text, max_line_chars=max_line_chars)
            merged.append(Cue(start=cur.start, end=nxt.end, text=combined_text, words=combined_words))
            i += 2
        else:
            merged.append(cur)
            i += 1
    return merged

