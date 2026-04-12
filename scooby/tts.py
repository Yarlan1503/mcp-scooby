"""ElevenLabs TTS engine for mcp-scooby.

Handles text-to-speech synthesis with:
    - Rate limiting (0.25s between requests)
    - Retry with exponential backoff (3 attempts)
    - Context stitching (previous_text / next_text)
    - Chunking for texts > 4500 chars
"""

import time
import logging
from typing import Optional

import requests

from .config import (
    DEFAULT_MODEL,
    DEFAULT_LANGUAGE_CODE,
    DEFAULT_OUTPUT_FORMAT,
    MAX_CHUNK_CHARS,
    MAX_RETRIES,
    RETRYABLE_STATUS,
    INITIAL_BACKOFF,
    REQUEST_DELAY,
)

logger = logging.getLogger(__name__)


def _split_text_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split long text into chunks at natural break points.

    Tries to cut at periods (.), then commas (,), then spaces.
    Each chunk will be at most ``max_chars`` characters.

    Args:
        text: Text to split.
        max_chars: Maximum characters per chunk.

    Returns:
        List of string chunks.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        segment = remaining[:max_chars]

        # Try cutting at period
        cut_pos = segment.rfind(".")
        if cut_pos > max_chars * 0.3:
            chunks.append(remaining[: cut_pos + 1])
            remaining = remaining[cut_pos + 1 :].lstrip()
            continue

        # Try cutting at comma
        cut_pos = segment.rfind(",")
        if cut_pos > max_chars * 0.3:
            chunks.append(remaining[: cut_pos + 1])
            remaining = remaining[cut_pos + 1 :].lstrip()
            continue

        # Try cutting at space
        cut_pos = segment.rfind(" ")
        if cut_pos > 0:
            chunks.append(remaining[:cut_pos])
            remaining = remaining[cut_pos:].lstrip()
            continue

        # Last resort: hard cut
        chunks.append(remaining[:max_chars])
        remaining = remaining[max_chars:].lstrip()

    return chunks


def _make_tts_request(url: str, headers: dict, body: dict, text_preview: str) -> bytes:
    """Execute a TTS request with retry and exponential backoff.

    Args:
        url: TTS endpoint URL.
        headers: HTTP headers.
        body: JSON request body.
        text_preview: Text preview for error messages.

    Returns:
        Audio bytes from successful response.

    Raises:
        RuntimeError: If all retries fail.
        ConnectionError: On unrecoverable connection errors.
    """
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.post(url, headers=headers, json=body, timeout=60)

            if response.status_code == 200:
                return response.content

            if response.status_code in RETRYABLE_STATUS:
                last_error = (
                    f"HTTP {response.status_code} (attempt {attempt}/{MAX_RETRIES})"
                )
                if attempt < MAX_RETRIES:
                    backoff = INITIAL_BACKOFF**attempt
                    logger.warning(
                        "Retry: %s — waiting %ds (text: '%s...')",
                        last_error,
                        backoff,
                        text_preview[:50],
                    )
                    time.sleep(backoff)
                continue

            # Non-retryable error
            detail = ""
            try:
                detail = response.json().get("detail", {})
            except Exception:
                detail = response.text[:200]
            raise RuntimeError(f"TTS error (HTTP {response.status_code}): {detail}")

        except requests.exceptions.Timeout:
            last_error = f"Timeout (attempt {attempt}/{MAX_RETRIES})"
            if attempt < MAX_RETRIES:
                backoff = INITIAL_BACKOFF**attempt
                logger.warning("Retry: %s — waiting %ds", last_error, backoff)
                time.sleep(backoff)
            continue

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"TTS connection error: {e}") from e

    raise RuntimeError(
        f"All retries failed for text: '{text_preview[:50]}...' — {last_error}"
    )


def tts_line(
    text: str,
    voice_id: str,
    api_key: str,
    model_id: str = DEFAULT_MODEL,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    previous_text: Optional[str] = None,
    next_text: Optional[str] = None,
) -> tuple[bytes, int]:
    """Synthesize a line of text using ElevenLabs TTS API.

    For texts exceeding MAX_CHUNK_CHARS, splits into chunks and makes
    multiple requests with stitching context.

    Args:
        text: Text to synthesize.
        voice_id: ElevenLabs voice ID.
        api_key: ElevenLabs API key.
        model_id: TTS model ID.
        language_code: Language code (e.g. "es").
        output_format: Output format (e.g. "mp3_44100_128").
        previous_text: Preceding text for context stitching.
        next_text: Following text for context stitching.

    Returns:
        Tuple of (audio_bytes, characters_used).

    Raises:
        RuntimeError: On TTS API errors after retries.
        ConnectionError: On connection failures.
    """
    chunks = _split_text_into_chunks(text)
    all_audio = b""
    total_chars = 0

    for chunk_idx, chunk in enumerate(chunks):
        # Context: previous_text only for first chunk, next_text only for last
        prev = previous_text if chunk_idx == 0 else ""
        nxt = next_text if chunk_idx == len(chunks) - 1 else ""

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        body = {
            "text": chunk,
            "model_id": model_id,
            "language_code": language_code,
            "output_format": output_format,
            "previous_text": prev or "",
            "next_text": nxt or "",
        }

        audio = _make_tts_request(url, headers, body, chunk)
        all_audio += audio
        total_chars += len(chunk)

        # Rate limit between chunks
        if chunk_idx < len(chunks) - 1:
            time.sleep(REQUEST_DELAY)

    return all_audio, total_chars
