"""ElevenLabs quota verification for mcp-scooby."""

import logging

import requests

from .config import get_api_key

logger = logging.getLogger(__name__)


def check_quota(api_key: str | None = None) -> dict:
    """Check ElevenLabs subscription quota.

    Args:
        api_key: API key (loaded from config if None).

    Returns:
        Dict with subscription info including:
            - tier: Subscription tier name
            - character_count: Characters used this period
            - character_limit: Character limit for the tier
            - characters_remaining: Limit minus used
            - estimated_reports_remaining: Approx reports remaining (~400 chars each)

    Raises:
        ConnectionError: On API connection failure.
        RuntimeError: On API error response.
    """
    if api_key is None:
        api_key = get_api_key()

    url = "https://api.elevenlabs.io/v1/user/subscription"
    headers = {"xi-api-key": api_key}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Cannot connect to ElevenLabs API: {e}") from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"ElevenLabs API error: {e}") from e

    data = response.json()

    char_count = data.get("character_count", 0)
    char_limit = data.get("character_limit", 0)
    remaining = char_limit - char_count

    return {
        "tier": data.get("tier", "unknown"),
        "characters_used": char_count,
        "characters_limit": char_limit,
        "characters_remaining": remaining,
        "estimated_reports_remaining": max(0, remaining // 400),
    }
