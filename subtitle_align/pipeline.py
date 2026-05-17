from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .align import (
    Segment,
    Word,
    build_segments_from_words,
    collect_words,
    detect_holes_from_segments,
    detect_holes_from_words,
    merge_words_replace_interval,
    segments_from_asr_jsonable,
    segments_to_jsonable,
    word_timestamp_coverage,
)
from .asr_faster_whisper import ASRResult, ASRSegment, transcribe_faster_whisper
from .config import RunConfig
from .ffmpeg_utils import (
    FFmpegError,
    check_ffmpeg_available,
    extract_audio_16k_mono_wav,
    extract_audio_16k_mono_wav_enhanced,
    extract_wav_segment,
    probe_duration_sec,
)
from .receipt import RunReceipt, join_degraded_reasons
from .subtitles import WordTS, build_cues_from_segments, build_cues_from_words, format_srt, format_vtt
from .utils import atomic_write_json, atomic_write_text, ensure_dir, format_exc, now_utc_iso, safe_rmtree
from .gates import (
    RemovedWordSample,
    apply_island_words_gate,
    apply_word_duration_gate,
    enforce_caption_duration_hard,
    load_hallucination_lexicon,
)
from .ocr_check import run_ocr_check
from .transcript_io import load_transcript


def run_pipeline(cfg: RunConfig) -> int:
    """
    End-to-end run:
    input -> ffmpeg wav (+ optional enhancement) -> primary ASR -> hole detection -> hole ASR -> merge -> SRT/VTT + transcript.json + run_receipt.json + logs.txt
    """
    ensure_dir(cfg.output_dir)
    logs_path = cfg.output_dir / "logs.txt"
    logger = _setup_logger(logs_path)

    started_at = time.time()
    warnings: list[str] = []
    errors: list[str] = []
    degraded_reasons: list[str] = []
    metrics: dict[str, Any] = {}
    versions: dict[str, Any] = {}
    receipt = RunReceipt(
        started_at_utc=now_utc_iso(),
        ended_at_utc=None,
        config=cfg.to_jsonable(),
        status="failed",
        degraded_reason=None,
        warnings=warnings,
        errors=errors,
        metrics=metrics,
        versions=versions,
    )

    temp_dir = cfg.output_dir / "_temp"
    try:
        # If user runs transcript-only mode without OCR, input media is not required.
        if not cfg.input_transcript_path or cfg.enable_ocr:
            if not cfg.input_path.exists():
                raise FileNotFoundError(f"input not found: {cfg.input_path}")

        versions.update(check_ffmpeg_available())

        safe_rmtree(temp_dir)
        ensure_dir(temp_dir)

        # Transcript-only mode: postprocess an existing transcript.json with gates/OCR and regenerate SRT/VTT.
        if cfg.input_transcript_path:
            return _run_postprocess_from_transcript(cfg, logger, receipt, temp_dir)

        wav_plain = temp_dir / "audio_16k_mono.wav"
        wav_enh = temp_dir / "audio_16k_mono_enhanced.wav"
        wav_path = wav_plain

        logger.info("ffmpeg: extracting 16kHz mono wav (enhance=%s)", cfg.enable_audio_enhance)
        if cfg.enable_audio_enhance:
            try:
                p = extract_audio_16k_mono_wav_enhanced(cfg.input_path, wav_enh)
                _log_ffmpeg(logger, p)
                wav_path = wav_enh
                metrics["audio_enhance"] = True
            except Exception as e:
                warnings.append("audio_enhance_failed_fallback_plain")
                logger.warning("Audio enhance failed, fallback to plain extract: %s", str(e)[:500])
                p = extract_audio_16k_mono_wav(cfg.input_path, wav_plain)
                _log_ffmpeg(logger, p)
                wav_path = wav_plain
                metrics["audio_enhance"] = False
        else:
            p = extract_audio_16k_mono_wav(cfg.input_path, wav_plain)
            _log_ffmpeg(logger, p)
            wav_path = wav_plain
            metrics["audio_enhance"] = False

        audio_duration = probe_duration_sec(cfg.input_path) or probe_duration_sec(wav_path)
        metrics["audio_duration_sec"] = audio_duration

        logger.info(
            "Primary ASR: model=%s compute_type=%s threads=%s language=%s vad_filter=%s beam=%s",
            cfg.model,
            cfg.compute_type,
            cfg.threads,
            cfg.language,
            cfg.vad_filter,
            cfg.beam_size,
        )
        asr_primary = transcribe_faster_whisper(
            wav_path,
            model_size=cfg.model,
            language=cfg.language,
            threads=cfg.threads,
            compute_type=cfg.compute_type,
            vad_filter=cfg.vad_filter,
            beam_size=cfg.beam_size,
            no_speech_threshold=cfg.no_speech_threshold,
            log_prob_threshold=cfg.log_prob_threshold,
            compression_ratio_threshold=cfg.compression_ratio_threshold,
        )
        metrics["asr_language"] = asr_primary.language
        metrics["language_probability"] = asr_primary.language_probability
        metrics["word_timestamps_supported"] = bool(asr_primary.word_timestamps_supported)

        segments_primary_json = _segments_to_jsonable(asr_primary.segments)
        segments_primary = segments_from_asr_jsonable(segments_primary_json)
        words_primary = collect_words(segments_primary, offset_sec=0.0)
        coverage_primary = word_timestamp_coverage(words_primary)
        metrics["word_timestamp_coverage_primary"] = coverage_primary
        metrics["num_words_primary"] = sum(1 for w in words_primary if w.start is not None and w.end is not None)

        holes = []
        if audio_duration and audio_duration > 0 and cfg.hole_gap_sec > 0:
            if asr_primary.word_timestamps_supported and coverage_primary > 0:
                holes = detect_holes_from_words(
                    words_primary,
                    duration_sec=float(audio_duration),
                    hole_gap_sec=cfg.hole_gap_sec,
                )
            else:
                holes = detect_holes_from_segments(
                    segments_primary,
                    duration_sec=float(audio_duration),
                    hole_gap_sec=cfg.hole_gap_sec,
                )
        metrics["num_gaps_over_threshold"] = len(holes)

        merged_words = words_primary
        hole_runs: list[dict[str, Any]] = []
        if holes and (audio_duration and audio_duration > 0):
            for i, h in enumerate(holes):
                hole_wav = temp_dir / f"hole_{i:03d}_{h.run_start:.2f}_{h.run_end:.2f}.wav"
                logger.info("Hole #%d: core=[%.2f, %.2f] run=[%.2f, %.2f]", i, h.core_start, h.core_end, h.run_start, h.run_end)
                try:
                    p = extract_wav_segment(wav_path, hole_wav, start_sec=h.run_start, end_sec=h.run_end)
                    _log_ffmpeg(logger, p)
                except FFmpegError as e:
                    warnings.append("ffmpeg_hole_cut_failed")
                    logger.warning("Hole cut failed: %s", str(e)[:500])
                    continue

                asr_hole = transcribe_faster_whisper(
                    hole_wav,
                    model_size=cfg.hole_pass_model,
                    language=cfg.language,
                    threads=cfg.threads,
                    compute_type=cfg.compute_type,
                    vad_filter=False,  # recall-first for holes
                    beam_size=cfg.hole_beam_size,
                    no_speech_threshold=cfg.no_speech_threshold,
                    log_prob_threshold=cfg.log_prob_threshold,
                    compression_ratio_threshold=cfg.compression_ratio_threshold,
                )
                hole_segments_json = _segments_to_jsonable(asr_hole.segments)
                hole_segments = segments_from_asr_jsonable(hole_segments_json)
                hole_words_abs = collect_words(hole_segments, offset_sec=h.run_start)
                merged_words = merge_words_replace_interval(
                    merged_words,
                    hole_words_abs,
                    start_sec=h.core_start,
                    end_sec=h.core_end,
                )
                hole_runs.append(
                    {
                        "index": i,
                        "core_start": h.core_start,
                        "core_end": h.core_end,
                        "run_start": h.run_start,
                        "run_end": h.run_end,
                        "model": cfg.hole_pass_model,
                        "beam_size": cfg.hole_beam_size,
                        "num_words": sum(1 for w in hole_words_abs if w.start is not None and w.end is not None),
                        "word_timestamps_supported": bool(asr_hole.word_timestamps_supported),
                    }
                )

        # Gates: anti-hallucination / non-speech filtering.
        halluc_lex = load_hallucination_lexicon(cfg.hallucination_lexicon_path)
        removed_samples: list[RemovedWordSample] = []

        merged_words, removed1 = apply_word_duration_gate(
            merged_words,
            max_word_dur=cfg.max_word_dur,
            removed_word_placeholder=cfg.removed_word_placeholder,
            hallucination_lexicon=halluc_lex,
        )
        removed_samples.extend(removed1)

        merged_words, removed2, island_warnings = apply_island_words_gate(
            merged_words,
            hole_gap_sec=cfg.hole_gap_sec,
            island_window_sec=cfg.island_window_sec,
            island_max_words=cfg.island_max_words,
            removed_word_placeholder=cfg.removed_word_placeholder,
            hallucination_lexicon=halluc_lex,
        )
        removed_samples.extend(removed2)
        warnings.extend(island_warnings)

        merged_coverage = word_timestamp_coverage(merged_words)
        metrics["word_timestamp_coverage"] = merged_coverage
        metrics["num_words"] = sum(1 for w in merged_words if w.start is not None and w.end is not None)
        metrics["hole_runs"] = hole_runs
        metrics["num_words_removed_by_gate"] = len(removed_samples)
        metrics["removed_samples"] = [s.to_jsonable() for s in removed_samples[:10]]

        if not asr_primary.word_timestamps_supported:
            degraded_reasons.append("no_word_timestamps")
        if merged_coverage < 0.98:
            warnings.append(f"low_word_timestamp_coverage<{0.98}: {merged_coverage:.4f}")
            degraded_reasons.append("low_word_timestamp_coverage")

        # Write outputs
        transcript_path = cfg.output_dir / "transcript.json"
        srt_path = cfg.output_dir / "subtitle.srt"
        vtt_path = cfg.output_dir / "subtitle.vtt"
        receipt_path = cfg.output_dir / "run_receipt.json"

        # Build merged segments from gated words for stable downstream usage.
        merged_segments = build_segments_from_words(merged_words)
        merged_segments_json = segments_to_jsonable(merged_segments)

        transcript = _build_transcript_json(
            cfg=cfg,
            asr_primary=asr_primary,
            wav_path=wav_path,
            audio_duration=audio_duration,
            segments_primary=segments_primary_json,
            segments_merged=merged_segments_json,
            hole_runs=hole_runs,
        )
        atomic_write_json(transcript_path, transcript)

        if asr_primary.word_timestamps_supported and merged_words:
            cues_words = [
                WordTS(text=w.word, start=w.start, end=w.end) for w in merged_words if w.start is not None and w.end is not None
            ]
            cues = build_cues_from_words(
                cues_words,
                gap_threshold=cfg.gap_threshold,
                max_line_chars=cfg.max_line_chars,
                min_duration=cfg.min_duration,
                max_duration=cfg.max_duration,
            )
        else:
            warnings.append("no_word_timestamps_using_segments_only")
            cues = build_cues_from_segments(segments_primary_json, max_line_chars=cfg.max_line_chars)
            degraded_reasons.append("no_word_timestamps")

        # Hard caption duration gate (force resegment/drop).
        cues, num_reseg, dur_warnings = enforce_caption_duration_hard(
            cues,
            max_caption_dur_hard=cfg.max_caption_dur_hard,
            gap_threshold=cfg.gap_threshold,
        )
        metrics["num_captions_resegmented"] = num_reseg
        warnings.extend(dur_warnings)

        # Optional OCR validation: never blocks the main pipeline.
        if cfg.enable_ocr:
            ocr_tmp = temp_dir / "ocr"
            cues2, ocr_warnings, ocr_metrics = run_ocr_check(
                cues,
                input_media=cfg.input_path,
                temp_dir=ocr_tmp,
                tesseract_cmd=cfg.tesseract_cmd,
                frames_per_caption=cfg.ocr_frames_per_caption,
                similarity_threshold=cfg.ocr_similarity_threshold,
            )
            cues = cues2
            warnings.extend(ocr_warnings)
            metrics["ocr"] = ocr_metrics

        atomic_write_text(srt_path, format_srt(cues))
        atomic_write_text(vtt_path, format_vtt(cues))

        ended_at = time.time()
        proc_time = ended_at - started_at
        receipt.ended_at_utc = now_utc_iso()
        metrics["processing_time_sec"] = proc_time
        if audio_duration and audio_duration > 0:
            metrics["rtf"] = proc_time / audio_duration

        metrics["model"] = cfg.model
        metrics["threads"] = cfg.threads
        metrics["device"] = cfg.device
        metrics["compute_type"] = cfg.compute_type
        metrics["vad_filter"] = cfg.vad_filter
        metrics["beam_size"] = cfg.beam_size
        metrics["no_speech_threshold"] = cfg.no_speech_threshold
        metrics["log_prob_threshold"] = cfg.log_prob_threshold
        metrics["compression_ratio_threshold"] = cfg.compression_ratio_threshold

        receipt.degraded_reason = join_degraded_reasons(degraded_reasons)
        if receipt.degraded_reason:
            receipt.status = "degraded"
        else:
            receipt.status = "success"

        atomic_write_json(receipt_path, receipt.to_jsonable())
        logger.info("Done. status=%s output=%s", receipt.status, cfg.output_dir)
        return 0 if receipt.status != "failed" else 2

    except Exception as e:
        errors.append(str(e))
        errors.append(format_exc(e))
        receipt.status = "failed"
        receipt.ended_at_utc = now_utc_iso()
        metrics["processing_time_sec"] = time.time() - started_at
        try:
            atomic_write_json(cfg.output_dir / "run_receipt.json", receipt.to_jsonable())
        except Exception:
            pass
        logger.exception("Pipeline failed: %s", e)
        return 2
    finally:
        if not cfg.keep_temp:
            safe_rmtree(temp_dir)


def _setup_logger(logs_path: Path) -> logging.Logger:
    logger = logging.getLogger("subtitle_align")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(str(logs_path), encoding="utf-8", mode="w")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _segments_to_jsonable(segments: list[ASRSegment]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in segments:
        out.append(
            {
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in s.words
                ],
            }
        )
    return out


def _build_transcript_json(
    *,
    cfg: RunConfig,
    asr_primary: ASRResult,
    wav_path: Path,
    audio_duration: float | None,
    segments_primary: list[dict[str, Any]],
    segments_merged: list[dict[str, Any]],
    hole_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "input": {"path": str(cfg.input_path)},
        "audio": {
            "wav_path": str(wav_path),
            "duration_sec": audio_duration,
            "enhanced": bool(cfg.enable_audio_enhance),
        },
        "language": asr_primary.language,
        "raw_text_primary": asr_primary.raw_text,
        "raw_text": "".join(s.get("text", "") for s in segments_merged).strip(),
        "asr": {
            "backend": asr_primary.backend,
            "model": cfg.model,
            "compute_type": cfg.compute_type,
            "requested_language": cfg.language,
            "detected_language": asr_primary.language,
            "language_probability": asr_primary.language_probability,
            "word_timestamps_supported": bool(asr_primary.word_timestamps_supported),
            "vad_filter": cfg.vad_filter,
            "beam_size": cfg.beam_size,
            "no_speech_threshold": cfg.no_speech_threshold,
            "log_prob_threshold": cfg.log_prob_threshold,
            "compression_ratio_threshold": cfg.compression_ratio_threshold,
        },
        "hole_filling": {
            "hole_gap_sec": cfg.hole_gap_sec,
            "hole_pass_model": cfg.hole_pass_model,
            "hole_beam_size": cfg.hole_beam_size,
            "runs": hole_runs,
        },
        "gates": {
            "max_word_dur": cfg.max_word_dur,
            "removed_word_placeholder": cfg.removed_word_placeholder,
            "island_window_sec": cfg.island_window_sec,
            "island_max_words": cfg.island_max_words,
            "max_caption_dur_hard": cfg.max_caption_dur_hard,
            "hallucination_lexicon_path": cfg.hallucination_lexicon_path,
        },
        "ocr": {
            "enabled": bool(cfg.enable_ocr),
            "frames_per_caption": cfg.ocr_frames_per_caption,
            "similarity_threshold": cfg.ocr_similarity_threshold,
        },
        "segments_primary": segments_primary,
        "segments": segments_merged,
    }


def _log_ffmpeg(logger: logging.Logger, p: Any) -> None:
    stderr = (getattr(p, "stderr", "") or "").strip()
    if stderr:
        logger.info("ffmpeg stderr:\n%s", stderr[:8000])


def _run_postprocess_from_transcript(cfg: RunConfig, logger: logging.Logger, receipt: RunReceipt, temp_dir: Path) -> int:
    warnings = receipt.warnings
    metrics = receipt.metrics

    tpath = Path(cfg.input_transcript_path or "")
    if not tpath.exists():
        raise FileNotFoundError(f"input_transcript not found: {tpath}")

    warnings.append("postprocess_from_transcript")
    lt = load_transcript(tpath)
    metrics["transcript_path"] = str(tpath)
    metrics["asr_language"] = lt.language

    words = lt.words
    metrics["num_words_before_gate"] = sum(1 for w in words if w.start is not None and w.end is not None)

    halluc_lex = load_hallucination_lexicon(cfg.hallucination_lexicon_path)
    removed_samples: list[RemovedWordSample] = []

    words, removed1 = apply_word_duration_gate(
        words,
        max_word_dur=cfg.max_word_dur,
        removed_word_placeholder=cfg.removed_word_placeholder,
        hallucination_lexicon=halluc_lex,
    )
    removed_samples.extend(removed1)

    words, removed2, island_warnings = apply_island_words_gate(
        words,
        hole_gap_sec=cfg.hole_gap_sec,
        island_window_sec=cfg.island_window_sec,
        island_max_words=cfg.island_max_words,
        removed_word_placeholder=cfg.removed_word_placeholder,
        hallucination_lexicon=halluc_lex,
    )
    removed_samples.extend(removed2)
    warnings.extend(island_warnings)

    metrics["num_words_removed_by_gate"] = len(removed_samples)
    metrics["removed_samples"] = [s.to_jsonable() for s in removed_samples[:10]]

    # Build cues from the (gated) words using existing segmentation rules.
    cues_words = [WordTS(text=w.word, start=w.start, end=w.end) for w in words if w.start is not None and w.end is not None]
    cues = build_cues_from_words(
        cues_words,
        gap_threshold=cfg.gap_threshold,
        max_line_chars=cfg.max_line_chars,
        min_duration=cfg.min_duration,
        max_duration=cfg.max_duration,
    )
    cues, num_reseg, dur_warnings = enforce_caption_duration_hard(
        cues,
        max_caption_dur_hard=cfg.max_caption_dur_hard,
        gap_threshold=cfg.gap_threshold,
    )
    metrics["num_captions_resegmented"] = num_reseg
    warnings.extend(dur_warnings)

    if cfg.enable_ocr:
        ocr_tmp = temp_dir / "ocr"
        cues2, ocr_warnings, ocr_metrics = run_ocr_check(
            cues,
            input_media=cfg.input_path,
            temp_dir=ocr_tmp,
            tesseract_cmd=cfg.tesseract_cmd,
            frames_per_caption=cfg.ocr_frames_per_caption,
            similarity_threshold=cfg.ocr_similarity_threshold,
        )
        cues = cues2
        warnings.extend(ocr_warnings)
        metrics["ocr"] = ocr_metrics

    # Write outputs (regenerated subtitles).
    atomic_write_text(cfg.output_dir / "subtitle.srt", format_srt(cues))
    atomic_write_text(cfg.output_dir / "subtitle.vtt", format_vtt(cues))

    # Also write a transcript.json reflecting the gated words/cues, keeping original for traceability.
    merged_segments = build_segments_from_words([Word(word=w.text, start=w.start, end=w.end) for w in cues_words])
    merged_segments_json = segments_to_jsonable(merged_segments)
    transcript_obj = {
        "input": {"path": str(cfg.input_path)},
        "source_transcript": str(tpath),
        "language": lt.language,
        "segments_primary": lt.segments,
        "segments": merged_segments_json,
    }
    atomic_write_json(cfg.output_dir / "transcript.json", transcript_obj)

    receipt.ended_at_utc = now_utc_iso()
    receipt.status = "success"
    atomic_write_json(cfg.output_dir / "run_receipt.json", receipt.to_jsonable())
    logger.info("Postprocess done. output=%s", cfg.output_dir)
    return 0
