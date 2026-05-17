from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class ASRWord:
    word: str
    start: Optional[float]
    end: Optional[float]
    probability: Optional[float]


@dataclass(frozen=True)
class ASRSegment:
    id: int
    start: float
    end: float
    text: str
    words: list[ASRWord]


@dataclass(frozen=True)
class ASRResult:
    language: str
    language_probability: float | None
    raw_text: str
    segments: list[ASRSegment]
    model: str
    backend: str = "faster-whisper"
    word_timestamps_supported: bool = True

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "language_probability": self.language_probability,
            "raw_text": self.raw_text,
            "word_timestamps_supported": self.word_timestamps_supported,
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in s.words
                    ],
                }
                for s in self.segments
            ],
        }


def transcribe_faster_whisper(
    audio_wav_path: Path,
    *,
    model_size: str,
    language: str,
    threads: int,
    compute_type: str = "int8",
    vad_filter: bool = False,
    beam_size: int = 5,
    no_speech_threshold: float = 0.3,
    log_prob_threshold: float = -2.0,
    compression_ratio_threshold: float = 3.0,
) -> ASRResult:
    """
    CPU-only ASR using faster-whisper (CTranslate2).

    Returns segments + (model-provided) word timestamps. WhisperX can optionally refine alignment later.
    """
    from faster_whisper import WhisperModel

    lang = None if language == "auto" else language
    model = WhisperModel(
        model_size_or_path=model_size,
        device="cpu",
        compute_type=compute_type,
        cpu_threads=max(1, int(threads)),
        num_workers=1,
    )

    kwargs: dict[str, Any] = {
        "language": lang,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "word_timestamps": True,
        "temperature": 0.0,
        "no_speech_threshold": no_speech_threshold,
        "log_prob_threshold": log_prob_threshold,
        "compression_ratio_threshold": compression_ratio_threshold,
    }
    segments_iter, info, word_ts_supported = _transcribe_with_compat(model, str(audio_wav_path), kwargs)

    segments: list[ASRSegment] = []
    raw_text_parts: list[str] = []

    for idx, seg in enumerate(_iter_segments(segments_iter)):
        raw_text_parts.append(str(seg.text))
        words: list[ASRWord] = []
        for w in getattr(seg, "words", []) or []:
            words.append(
                ASRWord(
                    word=str(getattr(w, "word", "")),
                    start=getattr(w, "start", None),
                    end=getattr(w, "end", None),
                    probability=getattr(w, "probability", None),
                )
            )
        segments.append(
            ASRSegment(
                id=idx,
                start=float(seg.start),
                end=float(seg.end),
                text=str(seg.text),
                words=words,
            )
        )

    detected_lang = getattr(info, "language", None) or (language if language != "auto" else "unknown")
    lang_prob = getattr(info, "language_probability", None)
    raw_text = "".join(raw_text_parts).strip()
    return ASRResult(
        language=str(detected_lang),
        language_probability=(float(lang_prob) if lang_prob is not None else None),
        raw_text=raw_text,
        segments=segments,
        model=model_size,
        word_timestamps_supported=word_ts_supported,
    )


def _iter_segments(segments_iter: Iterable[Any]) -> Iterable[Any]:
    for seg in segments_iter:
        yield seg


def _transcribe_with_compat(model: Any, audio_path: str, kwargs: dict[str, Any]) -> tuple[Iterable[Any], Any, bool]:
    """
    Handles faster-whisper version differences:
    - Some versions may not support word_timestamps or certain thresholds.
    - We retry by removing unsupported kwargs.
    """
    # First attempt: full kwargs (word timestamps on).
    try:
        segments_iter, info = model.transcribe(audio_path, **kwargs)
        return segments_iter, info, True
    except TypeError as e:
        msg = str(e)
        # If word_timestamps is unsupported, fallback to segment-only and mark degraded.
        if "word_timestamps" in msg and "unexpected keyword" in msg:
            kwargs2 = dict(kwargs)
            kwargs2.pop("word_timestamps", None)
            segments_iter, info = model.transcribe(audio_path, **kwargs2)
            return segments_iter, info, False

        # Remove possibly-unsupported thresholds and retry.
        retry_keys = ["no_speech_threshold", "log_prob_threshold", "compression_ratio_threshold"]
        kwargs2 = dict(kwargs)
        removed = False
        for k in retry_keys:
            if k in kwargs2:
                kwargs2.pop(k, None)
                removed = True
        if removed:
            segments_iter, info = model.transcribe(audio_path, **kwargs2)
            # word timestamps still expected if not removed
            return segments_iter, info, bool(kwargs.get("word_timestamps", False))
        raise
