from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import RunConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m subtitle_align",
        description="Windows CPU-only subtitle generator (English-friendly): faster-whisper word timestamps + audio enhance + automatic hole-filling.",
    )
    parser.add_argument("--input", required=True, help="Input media file (mp4/mkv/mov/mp3/wav).")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--input_transcript",
        default="",
        help="Optional: use an existing transcript.json to postprocess with gates/OCR and regenerate SRT/VTT without ASR.",
    )
    parser.add_argument("--language", default="auto", help="auto|en|... (passed to ASR).")
    parser.add_argument("--model", default="small", choices=["small", "medium"], help="Whisper model size.")
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="Device (CPU-only).")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8, help="CPU threads for inference.")
    parser.add_argument("--compute_type", default="int8", choices=["int8", "float32"], help="CTranslate2 compute type.")

    parser.add_argument("--vad_filter", default="off", choices=["on", "off"], help="Voice activity detection filter.")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size.")
    parser.add_argument("--no_speech_threshold", type=float, default=0.3, help="No-speech threshold (lower => more recall).")
    parser.add_argument("--log_prob_threshold", type=float, default=-2.0, help="Log prob threshold (lower => more recall).")
    parser.add_argument(
        "--compression_ratio_threshold",
        type=float,
        default=3.0,
        help="Compression ratio threshold (higher => more recall).",
    )

    parser.add_argument(
        "--enable_audio_enhance",
        default="on",
        choices=["on", "off"],
        help="Apply loudnorm+compressor+limiter to improve low-volume speech.",
    )
    parser.add_argument("--hole_gap_sec", type=float, default=3.0, help="Detect gaps larger than this and re-run ASR.")
    parser.add_argument("--hole_pass_model", default="small", choices=["small", "medium"], help="Model for hole pass.")
    parser.add_argument("--hole_beam_size", type=int, default=8, help="Beam size for hole pass.")

    parser.add_argument("--gap_threshold", type=float, default=0.35, help="Silence gap threshold (seconds).")
    parser.add_argument("--max_line_chars", type=int, default=42, help="Max chars per line.")
    parser.add_argument("--min_duration", type=float, default=1.0, help="Minimum cue duration (seconds).")
    parser.add_argument("--max_duration", type=float, default=4.5, help="Maximum cue duration (seconds).")

    parser.add_argument("--max_word_dur", type=float, default=1.2, help="Drop words with duration > this (anti-hallucination).")
    parser.add_argument(
        "--removed_word_placeholder",
        default="blank",
        choices=["blank", "ellipsis"],
        help="When a word is removed, keep blank timeline or insert a short '…' placeholder.",
    )
    parser.add_argument(
        "--max_caption_dur_hard",
        type=float,
        default=6.0,
        help="Hard cap for a single cue duration; longer cues are force resegmented or dropped.",
    )
    parser.add_argument("--island_window_sec", type=float, default=4.0, help="Island window size for island-words gate.")
    parser.add_argument("--island_max_words", type=int, default=2, help="Max words in an island to be removed.")
    parser.add_argument(
        "--hallucination_lexicon",
        default="",
        help="Optional lexicon file (.txt/.json) for frequent hallucination words (working/okay/yeah/...).",
    )

    parser.add_argument("--enable_ocr", default="off", choices=["on", "off"], help="Optional OCR validation via tesseract (burned-in subtitles).")
    parser.add_argument("--ocr_similarity_threshold", type=float, default=0.2, help="Mismatch threshold for OCR vs ASR.")
    parser.add_argument("--ocr_frames_per_caption", type=int, default=1, help="Frames per caption for OCR (1 or 2).")
    parser.add_argument("--tesseract_cmd", default="tesseract", help="Tesseract executable name/path.")

    parser.add_argument("--keep_temp", default="off", choices=["on", "off"], help="Keep temporary wav files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RunConfig(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        input_transcript_path=(str(args.input_transcript).strip() or None),
        language=args.language,
        model=args.model,
        device=args.device,
        threads=args.threads,
        compute_type=args.compute_type,
        vad_filter=(args.vad_filter == "on"),
        beam_size=int(args.beam_size),
        no_speech_threshold=float(args.no_speech_threshold),
        log_prob_threshold=float(args.log_prob_threshold),
        compression_ratio_threshold=float(args.compression_ratio_threshold),
        enable_audio_enhance=(args.enable_audio_enhance == "on"),
        hole_gap_sec=float(args.hole_gap_sec),
        hole_pass_model=args.hole_pass_model,
        hole_beam_size=int(args.hole_beam_size),
        gap_threshold=args.gap_threshold,
        max_line_chars=args.max_line_chars,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_word_dur=float(args.max_word_dur),
        removed_word_placeholder=str(args.removed_word_placeholder),
        island_window_sec=float(args.island_window_sec),
        island_max_words=int(args.island_max_words),
        max_caption_dur_hard=float(args.max_caption_dur_hard),
        hallucination_lexicon_path=(str(args.hallucination_lexicon).strip() or None),
        enable_ocr=(args.enable_ocr == "on"),
        ocr_similarity_threshold=float(args.ocr_similarity_threshold),
        ocr_frames_per_caption=int(args.ocr_frames_per_caption),
        tesseract_cmd=str(args.tesseract_cmd),
        keep_temp=(args.keep_temp == "on"),
    )
    return run_pipeline(config)
