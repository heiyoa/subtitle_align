from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .ffmpeg_utils import run_cmd
from .subtitles import Cue
from .utils import levenshtein_distance, word_count


@dataclass(frozen=True)
class OcrResult:
    text: str
    ok: bool
    error: str | None


def run_ocr_check(
    cues: list[Cue],
    *,
    input_media: Path,
    temp_dir: Path,
    tesseract_cmd: str,
    frames_per_caption: int,
    similarity_threshold: float,
) -> tuple[list[Cue], list[str], dict[str, Any]]:
    """
    Optional OCR/visual validation for burned-in subtitles.
    - Extracts 1-2 frames per cue with ffmpeg.
    - Runs tesseract CLI (free) to read on-screen text.
    - If OCR is stable and similarity is low, mark warning.
    - If OCR is stable and texts are essentially identical, can conservatively replace cue text.

    OCR failures never fail the pipeline.
    """
    warnings: list[str] = []
    metrics: dict[str, Any] = {"ocr_checked": 0, "ocr_mismatch": 0, "ocr_replaced": 0, "ocr_skipped": 0}
    out: list[Cue] = []

    if frames_per_caption not in (1, 2):
        frames_per_caption = 1

    if not _tesseract_available(tesseract_cmd):
        warnings.append("ocr_unavailable_tesseract_not_found")
        return cues, warnings, metrics

    temp_dir.mkdir(parents=True, exist_ok=True)
    for idx, c in enumerate(cues):
        metrics["ocr_checked"] += 1
        ocr_text = _ocr_for_cue(
            idx,
            input_media=input_media,
            temp_dir=temp_dir,
            tesseract_cmd=tesseract_cmd,
            cue=c,
            frames_per_caption=frames_per_caption,
        )
        if not ocr_text.ok:
            metrics["ocr_skipped"] += 1
            if ocr_text.error:
                warnings.append(f"ocr_failed cue={idx} err={ocr_text.error[:200]}")
            out.append(c)
            continue

        asr = c.text.replace("\n", " ").strip()
        ocr = ocr_text.text.replace("\n", " ").strip()
        if len(ocr) < 3:
            metrics["ocr_skipped"] += 1
            out.append(c)
            continue

        sim = similarity(asr, ocr)
        if sim < similarity_threshold:
            metrics["ocr_mismatch"] += 1
            warnings.append(f"low_confidence_visual_mismatch cue={idx} sim={sim:.3f} asr={_short(asr)} ocr={_short(ocr)}")
            out.append(c)
            continue

        replaced = _maybe_replace_text(asr, ocr)
        if replaced is not None and replaced != asr:
            metrics["ocr_replaced"] += 1
            out.append(Cue(start=c.start, end=c.end, text=replaced, words=c.words))
        else:
            out.append(c)

    return out, warnings, metrics


def similarity(a: str, b: str) -> float:
    a2 = _normalize_text(a)
    b2 = _normalize_text(b)
    if not a2 and not b2:
        return 1.0
    if not a2 or not b2:
        return 0.0
    lev = 1.0 - (levenshtein_distance(a2, b2) / max(1, max(len(a2), len(b2))))
    jac = _token_jaccard(a2, b2)
    return max(lev, jac)


def _maybe_replace_text(asr: str, ocr: str) -> Optional[str]:
    """
    Conservative replacement:
    - If normalized texts are identical (ignoring punctuation/case), use OCR text (may have better casing/punct).
    - Otherwise do not replace (only validate).
    """
    a = _normalize_text(asr)
    b = _normalize_text(ocr)
    if a == b and a:
        # Ensure word count doesn't drift in weird OCR cases.
        if abs(word_count(asr) - word_count(ocr)) <= 0:
            return ocr.strip()
    return None


def _ocr_for_cue(
    idx: int,
    *,
    input_media: Path,
    temp_dir: Path,
    tesseract_cmd: str,
    cue: Cue,
    frames_per_caption: int,
) -> OcrResult:
    times = _pick_times(cue.start, cue.end, frames_per_caption)
    texts: list[str] = []
    for j, t in enumerate(times):
        img = temp_dir / f"ocr_{idx:05d}_{j}_{t:.2f}.png"
        ok = _extract_frame(input_media, img, t)
        if not ok:
            return OcrResult(text="", ok=False, error="ffmpeg_extract_frame_failed")
        txt = _run_tesseract(tesseract_cmd, img)
        if txt:
            texts.append(txt)
    if not texts:
        return OcrResult(text="", ok=False, error="ocr_empty")
    if len(texts) == 1:
        return OcrResult(text=texts[0], ok=True, error=None)
    # 2 frames: require stability
    s = similarity(texts[0], texts[1])
    if s < 0.6:
        return OcrResult(text="", ok=False, error="ocr_unstable")
    return OcrResult(text=texts[0], ok=True, error=None)


def _extract_frame(input_media: Path, output_png: Path, t: float) -> bool:
    # Keep it cheap; scale down for OCR speed.
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(input_media),
        "-frames:v",
        "1",
        "-vf",
        "scale=iw*0.5:ih*0.5",
        str(output_png),
    ]
    p = run_cmd(cmd)
    return p.returncode == 0 and output_png.exists()


def _run_tesseract(tesseract_cmd: str, image_path: Path) -> str:
    try:
        p = subprocess.run(
            [tesseract_cmd, str(image_path), "stdout", "-l", "eng", "--psm", "6"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
            timeout=30,
        )
    except Exception:
        return ""
    if p.returncode != 0:
        return ""
    return (p.stdout or "").strip()


def _tesseract_available(tesseract_cmd: str) -> bool:
    try:
        p = subprocess.run(
            [tesseract_cmd, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
            timeout=10,
        )
    except Exception:
        return False
    return p.returncode == 0


def _pick_times(start: float, end: float, n: int) -> list[float]:
    if end <= start:
        return [start]
    if n <= 1:
        return [start + (end - start) * 0.5]
    return [start + (end - start) * 0.33, start + (end - start) * 0.66]


_NORM = re.compile(r"[^a-z0-9']+")


def _normalize_text(t: str) -> str:
    t = (t or "").lower()
    t = _NORM.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _token_jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _short(t: str) -> str:
    t = t.replace("\n", " ").strip()
    return t[:80] + ("…" if len(t) > 80 else "")

