"""Configuration: voices, defaults, paths for mcp-scooby."""

import os
from pathlib import Path
from typing import Dict

# --- Voices ---
VOICES: Dict[str, dict] = {
    "Mario": {
        "voice_id": "tomkxGQGz4b1kE0EM722",
        "accent": "latinoamericano",
        "style": "conversacional",
    },
    "Sara": {
        "voice_id": "gD1IexrzCvsXPHUuT0s3",
        "accent": "peninsular",
        "style": "conversacional",
    },
    "Javier": {
        "voice_id": "1vLlJCWRhRcfmTewn4cm",
        "accent": "peninsular",
        "style": "profundo",
    },
    "Enrique": {
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "accent": "británico",
        "style": "storyteller",
    },
}

# --- Defaults ---
DEFAULT_MODEL = "eleven_multilingual_v2"
DEFAULT_SPEAKER = "Mario"
DEFAULT_LANGUAGE_CODE = "es"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_PAUSE_SPEAKERS_MS = 800
DEFAULT_PAUSE_LINES_MS = 300
DEFAULT_SPEED = 1.25

# Heading pause durations in milliseconds
HEADING_PAUSES = {
    1: 1200,  # h1 → 1.2s
    2: 800,  # h2 → 0.8s
    3: 500,  # h3 → 0.5s
}

# --- Paths ---
FFMPEG_PATH = os.path.expanduser("~/.local/bin/ffmpeg")
API_KEY_FILE = os.path.expanduser("~/.config/secrets/elevenlabs.env")
API_KEY_ENV = "ELEVENLABS_API_KEY"

# --- TTS Limits ---
MAX_CHUNK_CHARS = 4500
MAX_RETRIES = 3
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
INITIAL_BACKOFF = 2  # seconds
REQUEST_DELAY = 0.25  # seconds between requests


def get_api_key() -> str:
    """Load ElevenLabs API key from env or secret file.

    Resolution order:
        1. Environment variable ELEVENLABS_API_KEY
        2. ~/.config/secrets/elevenlabs.env file

    Returns:
        API key string.

    Raises:
        ValueError: If key not found in any source.
    """
    # 1. From env var
    key = os.environ.get(API_KEY_ENV)
    if key:
        return key

    # 2. From secret file
    env_file = Path(API_KEY_FILE)
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(API_KEY_ENV + "="):
                return line.split("=", 1)[1].strip()

    raise ValueError(f"{API_KEY_ENV} not found in env or {API_KEY_FILE}")
