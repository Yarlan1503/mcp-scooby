"""Audio assembly for mcp-scooby.

Handles:
    - Silence generation for pauses
    - Segment concatenation via FFmpeg
    - Final MP3 output at 128kbps 44.1kHz
"""

import os
import shutil
import tempfile
import logging
import subprocess
from typing import Optional

from .config import FFMPEG_PATH, HEADING_PAUSES

logger = logging.getLogger(__name__)


def get_ffmpeg_path() -> str:
    """Locate FFmpeg binary in PATH or fallback location.

    Returns:
        Absolute path to FFmpeg binary.

    Raises:
        FileNotFoundError: If FFmpeg not found.
    """
    # Check PATH first
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    # Fallback
    if os.path.isfile(FFMPEG_PATH) and os.access(FFMPEG_PATH, os.X_OK):
        return FFMPEG_PATH

    raise FileNotFoundError(
        f"FFmpeg not found in PATH or at {FFMPEG_PATH}. "
        "Install FFmpeg or update FFMPEG_PATH in config."
    )


def generate_silence(
    duration_ms: int,
    output_path: str,
    ffmpeg_path: Optional[str] = None,
) -> str:
    """Generate an MP3 file of silence using FFmpeg.

    Args:
        duration_ms: Duration in milliseconds.
        output_path: Where to save the MP3.
        ffmpeg_path: Path to FFmpeg binary (auto-detected if None).

    Returns:
        Path to the generated file.

    Raises:
        RuntimeError: If FFmpeg fails.
    """
    if ffmpeg_path is None:
        ffmpeg_path = get_ffmpeg_path()

    seconds = duration_ms / 1000.0
    cmd = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r=44100:cl=mono",
        "-t",
        str(seconds),
        "-acodec",
        "libmp3lame",
        "-b:a",
        "128k",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg silence generation failed (code {result.returncode}): "
            f"{result.stderr}"
        )
    return output_path


def concatenate_audio(
    segment_paths: list[str],
    output_path: str,
    ffmpeg_path: Optional[str] = None,
) -> None:
    """Concatenate multiple audio files into one using FFmpeg.

    Uses FFmpeg concat demuxer for clean joining without re-encoding.

    Args:
        segment_paths: List of audio file paths to concatenate.
        output_path: Final output file path.
        ffmpeg_path: Path to FFmpeg binary (auto-detected if None).

    Raises:
        ValueError: If segment_paths is empty.
        RuntimeError: If FFmpeg fails.
    """
    if not segment_paths:
        raise ValueError("No audio segments to concatenate.")

    if ffmpeg_path is None:
        ffmpeg_path = get_ffmpeg_path()

    # Create concat list file
    list_fd, list_path = tempfile.mkstemp(suffix=".txt", prefix="scooby_concat_")
    try:
        with os.fdopen(list_fd, "w", encoding="utf-8") as f:
            for seg_path in segment_paths:
                # Escape for FFmpeg concat format
                escaped = seg_path.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        cmd = [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg concatenation failed (code {result.returncode}): "
                f"{result.stderr}"
            )
    finally:
        if os.path.exists(list_path):
            os.unlink(list_path)


def get_audio_duration(file_path: str, ffmpeg_path: Optional[str] = None) -> float:
    """Get duration of an audio file in seconds using FFmpeg ffprobe.

    Args:
        file_path: Path to the audio file.
        ffmpeg_path: Path to FFmpeg directory (ffprobe assumed alongside).

    Returns:
        Duration in seconds.
    """
    if ffmpeg_path is None:
        ffmpeg_path = get_ffmpeg_path()

    ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
    if not os.path.isfile(ffprobe_path):
        # Try alongside
        ffprobe_dir = os.path.dirname(ffmpeg_path)
        ffprobe_path = os.path.join(ffprobe_dir, "ffprobe")

    cmd = [
        ffprobe_path,
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass

    # Fallback: estimate from file size (128kbps MP3)
    file_size = os.path.getsize(file_path)
    return (file_size * 8) / (128 * 1000)
