#!/usr/bin/env python3
"""mcp-scooby — MCP server for Markdown-to-podcast audio via ElevenLabs TTS.

Tools:
    - text_to_speech: Convert Markdown file to MP3 audio
    - list_voices: List available voices with metadata
    - check_quota: Check ElevenLabs subscription quota
    - preview_speaker: Generate a short test clip for a speaker
"""

import os
import shutil
import tempfile
import time
import logging
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from scooby.config import (
    VOICES,
    DEFAULT_MODEL,
    DEFAULT_SPEAKER,
    DEFAULT_PAUSE_SPEAKERS_MS,
    HEADING_PAUSES,
    get_api_key,
)
from scooby.parser import parse_markdown
from scooby.tts import tts_line
from scooby.audio import generate_silence, concatenate_audio, get_audio_duration
from scooby.quota import check_quota as _check_quota

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MCP Server ---
mcp = FastMCP("scooby")


@mcp.tool()
def text_to_speech(
    markdown_path: str,
    output_path: Optional[str] = None,
    speakers: Optional[dict] = None,
    default_speaker: str = DEFAULT_SPEAKER,
    model: str = DEFAULT_MODEL,
    speed: float = 1.0,
    pause_speakers_ms: int = DEFAULT_PAUSE_SPEAKERS_MS,
    include_headings: bool = True,
) -> dict:
    """Convert a Markdown file to audio MP3 using ElevenLabs TTS.

    Parses Markdown with [Speaker] tags, generates multi-voice audio,
    inserts structural pauses for headings, and outputs a single MP3.

    Args:
        markdown_path: Path to the input .md file.
        output_path: Output MP3 path (default: same name as input, .mp3 extension).
        speakers: Optional override mapping of speaker_name → voice_id.
        default_speaker: Default speaker for lines without a tag.
        model: TTS model ID (default: eleven_multilingual_v2).
        speed: Playback speed 0.25-4.0 (default: 1.0).
        pause_speakers_ms: Pause between different speakers in ms.
        include_headings: Whether to read headings aloud.

    Returns:
        Dict with output_path, duration_seconds, characters_used,
        speakers_used, and lines_processed.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If no dialogue lines found or API key missing.
        RuntimeError: On TTS or FFmpeg errors.
    """
    # Validate input
    md_path = Path(markdown_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Input file not found: {markdown_path}")

    if not md_path.is_file():
        raise ValueError(f"Not a file: {markdown_path}")

    # Determine output path
    if output_path is None:
        output_path = str(md_path.with_suffix(".mp3"))

    # Load API key
    api_key = get_api_key()

    # Build voice mapping: merge defaults with overrides
    voice_map = {name: v["voice_id"] for name, v in VOICES.items()}
    if speakers:
        voice_map.update(speakers)

    # Read and parse markdown
    content = md_path.read_text(encoding="utf-8")
    lines = parse_markdown(
        content,
        default_speaker=default_speaker,
        include_headings=include_headings,
    )

    if not lines:
        raise ValueError("No dialogue lines found in the Markdown file.")

    logger.info(
        "Processing %d lines with %d speakers...",
        len(lines),
        len(set(l.speaker for l in lines)),
    )

    # Validate speakers
    for line in lines:
        if line.speaker not in voice_map:
            raise ValueError(
                f"Speaker '{line.speaker}' not found in voice mapping. "
                f"Available: {', '.join(voice_map.keys())}"
            )

    # Temp directory for segments
    tmp_dir = tempfile.mkdtemp(prefix="scooby_")

    try:
        segment_paths: list[str] = []
        total_chars = 0
        speakers_used = set()

        # Pre-generate silence files
        pause_files: dict[int, str] = {}
        for level, ms in HEADING_PAUSES.items():
            pause_path = os.path.join(tmp_dir, f"pause_h{level}.mp3")
            generate_silence(ms, pause_path)
            pause_files[level] = pause_path

        if pause_speakers_ms > 0:
            speaker_pause_path = os.path.join(tmp_dir, "pause_speakers.mp3")
            generate_silence(pause_speakers_ms, speaker_pause_path)

        line_pause_path = os.path.join(tmp_dir, "pause_lines.mp3")
        generate_silence(300, line_pause_path)

        # Process each dialogue line
        for i, line in enumerate(lines):
            voice_id = voice_map[line.speaker]
            speakers_used.add(line.speaker)

            # Add heading pause if this line has a heading
            if line.heading_level > 0 and line.heading_level in pause_files:
                segment_paths.append(pause_files[line.heading_level])
            elif i > 0 and line.heading_level > 0 and 3 in pause_files:
                # Fallback to h3 pause for any heading
                segment_paths.append(pause_files[3])

            # Add speaker transition pause
            if i > 0 and line.speaker != lines[i - 1].speaker:
                if pause_speakers_ms > 0:
                    segment_paths.append(speaker_pause_path)
            elif i > 0:
                segment_paths.append(line_pause_path)

            # Context stitching
            prev_text = lines[i - 1].text if i > 0 else None
            next_text = lines[i + 1].text if i < len(lines) - 1 else None

            logger.info(
                "[%d/%d] %s: %s...", i + 1, len(lines), line.speaker, line.text[:40]
            )

            # Generate TTS
            audio_bytes, chars = tts_line(
                text=line.text,
                voice_id=voice_id,
                api_key=api_key,
                model_id=model,
                previous_text=prev_text,
                next_text=next_text,
            )
            total_chars += chars

            # Save segment
            seg_path = os.path.join(tmp_dir, f"seg_{i:04d}.mp3")
            with open(seg_path, "wb") as f:
                f.write(audio_bytes)
            segment_paths.append(seg_path)

        # Concatenate all segments
        logger.info("Concatenating %d segments...", len(segment_paths))
        concatenate_audio(segment_paths, output_path)

        # Calculate duration
        duration = get_audio_duration(output_path)

        result = {
            "output_path": str(output_path),
            "duration_seconds": round(duration, 1),
            "characters_used": total_chars,
            "speakers_used": sorted(speakers_used),
            "lines_processed": len(lines),
        }

        logger.info(
            "Audio generated: %s (%.1fs, %d chars)", output_path, duration, total_chars
        )
        return result

    finally:
        # Cleanup temp directory
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


@mcp.tool()
def list_voices() -> list[dict]:
    """List available voices with their IDs and metadata.

    Returns:
        List of voice dicts with name, voice_id, accent, and style.
    """
    return [
        {
            "name": name,
            "voice_id": info["voice_id"],
            "accent": info["accent"],
            "style": info["style"],
        }
        for name, info in VOICES.items()
    ]


@mcp.tool()
def check_quota() -> dict:
    """Check remaining ElevenLabs subscription quota.

    Returns:
        Dict with tier, characters_used, characters_limit,
        characters_remaining, and estimated_reports_remaining.
    """
    return _check_quota()


@mcp.tool()
def preview_speaker(
    speaker_name: str,
    sample_text: Optional[str] = None,
    output_path: Optional[str] = None,
) -> dict:
    """Generate a short test clip for a speaker.

    Args:
        speaker_name: Name of the speaker to preview.
        sample_text: Custom text for the preview.
            Default: "Hola, soy {speaker_name}. Esta es una prueba de audio."
        output_path: Where to save the MP3. Default: /tmp/scooby_preview_{speaker}.mp3

    Returns:
        Dict with output_path and characters_used.

    Raises:
        ValueError: If speaker_name not found in voice mapping.
    """
    if speaker_name not in VOICES:
        raise ValueError(
            f"Speaker '{speaker_name}' not found. Available: {', '.join(VOICES.keys())}"
        )

    voice_id = VOICES[speaker_name]["voice_id"]
    api_key = get_api_key()

    if sample_text is None:
        sample_text = f"Hola, soy {speaker_name}. Esta es una prueba de audio."

    if output_path is None:
        output_path = f"/tmp/scooby_preview_{speaker_name.lower()}.mp3"

    audio_bytes, chars = tts_line(
        text=sample_text,
        voice_id=voice_id,
        api_key=api_key,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    return {
        "output_path": output_path,
        "characters_used": chars,
    }


if __name__ == "__main__":
    mcp.run()
