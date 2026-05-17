from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


class FFmpegError(RuntimeError):
    pass


def run_cmd(cmd: list[str], timeout_sec: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
        timeout=timeout_sec,
    )


def check_ffmpeg_available() -> dict:
    ffmpeg = run_cmd(["ffmpeg", "-version"])
    ffprobe = run_cmd(["ffprobe", "-version"])
    if ffmpeg.returncode != 0 or ffprobe.returncode != 0:
        raise FFmpegError(
            "ffmpeg/ffprobe not available in PATH. Install ffmpeg and ensure ffmpeg+ffprobe are callable."
        )
    return {
        "ffmpeg_version": (ffmpeg.stdout.splitlines() or ffmpeg.stderr.splitlines() or ["unknown"])[0].strip(),
        "ffprobe_version": (ffprobe.stdout.splitlines() or ffprobe.stderr.splitlines() or ["unknown"])[0].strip(),
    }


def probe_duration_sec(media_path: Path) -> Optional[float]:
    p = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
    )
    if p.returncode != 0:
        return None
    s = (p.stdout or "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract_audio_16k_mono_wav(input_path: Path, output_wav: Path) -> subprocess.CompletedProcess[str]:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_wav),
    ]
    p = run_cmd(cmd)
    if p.returncode != 0:
        raise FFmpegError(f"ffmpeg failed (code={p.returncode}). stderr:\n{p.stderr}")
    return p


def extract_audio_16k_mono_wav_enhanced(input_path: Path, output_wav: Path) -> subprocess.CompletedProcess[str]:
    """
    Extracts 16kHz mono wav and applies speech-friendly enhancement.
    If this fails, caller should fallback to extract_audio_16k_mono_wav.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    af = (
        "loudnorm=I=-16:TP=-1.5:LRA=11,"
        "acompressor=threshold=-18dB:ratio=4:attack=5:release=50,"
        "alimiter=limit=-1.0"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        af,
        "-f",
        "wav",
        str(output_wav),
    ]
    p = run_cmd(cmd)
    if p.returncode != 0:
        raise FFmpegError(f"ffmpeg enhanced extract failed (code={p.returncode}). stderr:\n{p.stderr}")
    return p


def extract_wav_segment(input_wav: Path, output_wav: Path, *, start_sec: float, end_sec: float) -> subprocess.CompletedProcess[str]:
    """
    Cut a wav segment [start_sec, end_sec] into another wav.
    """
    if end_sec <= start_sec:
        raise ValueError("end_sec must be > start_sec")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(input_wav),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_wav),
    ]
    p = run_cmd(cmd)
    if p.returncode != 0:
        raise FFmpegError(f"ffmpeg segment cut failed (code={p.returncode}). stderr:\n{p.stderr}")
    return p
