from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    input_path: Path
    output_dir: Path
    input_transcript_path: str | None = None

    language: str = "auto"
    model: str = "small"
    device: str = "cpu"
    threads: int = 8
    compute_type: str = "int8"

    vad_filter: bool = False
    beam_size: int = 5
    no_speech_threshold: float = 0.3
    log_prob_threshold: float = -2.0
    compression_ratio_threshold: float = 3.0

    enable_audio_enhance: bool = True
    hole_gap_sec: float = 3.0
    hole_pass_model: str = "small"
    hole_beam_size: int = 8

    # Anti-hallucination / non-speech gates
    max_word_dur: float = 1.2
    removed_word_placeholder: str = "blank"  # blank|ellipsis
    island_window_sec: float = 4.0
    island_max_words: int = 2
    max_caption_dur_hard: float = 6.0
    hallucination_lexicon_path: str | None = None

    # Optional OCR validation (free module via tesseract CLI)
    enable_ocr: bool = False
    ocr_similarity_threshold: float = 0.2
    ocr_frames_per_caption: int = 1
    tesseract_cmd: str = "tesseract"

    gap_threshold: float = 0.35
    max_line_chars: int = 42
    min_duration: float = 1.0
    max_duration: float = 4.5

    keep_temp: bool = False

    def to_jsonable(self) -> dict:
        d = asdict(self)
        d["input_path"] = str(self.input_path)
        d["output_dir"] = str(self.output_dir)
        return d
