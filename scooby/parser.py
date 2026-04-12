"""Markdown → structured dialogue parser for mcp-scooby.

Supports:
    - [Speaker] text → dialogue line
    - # / ## / ### headings → structural pauses (heading_level 1-3)
    - - item / * item → list items attributed to current speaker
    - **bold**, *italic*, ~~strikethrough~~ → stripped to plain text
    - [link text](url) → only "link text"
    - Lines without speaker → use last active speaker or default
    - Empty lines → ignored
    - Code blocks (```...```) → content as narration by current speaker
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DialogueLine:
    """A single parsed dialogue line."""

    speaker: str
    text: str  # Clean text (no markdown)
    heading_before: Optional[str] = None  # Heading text if a header preceded this
    heading_level: int = 0  # 0=no heading, 1=h1, 2=h2, 3=h3
    is_list_item: bool = False


# --- Regex patterns ---
SPEAKER_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
LIST_RE = re.compile(r"^[-*]\s+(.+)$")
CODE_FENCE_RE = re.compile(r"^```")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
ITALIC_RE = re.compile(r"\*(.+?)\*")
STRIKETHROUGH_RE = re.compile(r"~~(.+?)~~")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")


def clean_markdown(text: str) -> str:
    """Strip markdown formatting from text.

    Removes: bold (**...**), italic (*...*), strikethrough (~~...~~),
    links [text](url) → text, images ![alt](url) → ''.

    Args:
        text: Raw text potentially containing markdown.

    Returns:
        Clean plain text.
    """
    # Images first (before links, since ![...] contains [...])
    text = IMAGE_RE.sub("", text)
    # Links: [text](url) → text
    text = LINK_RE.sub(r"\1", text)
    # Bold: **text** → text
    text = BOLD_RE.sub(r"\1", text)
    # Italic: *text* → text
    text = ITALIC_RE.sub(r"\1", text)
    # Strikethrough: ~~text~~ → text
    text = STRIKETHROUGH_RE.sub(r"\1", text)
    # Extra whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_markdown(
    content: str,
    default_speaker: str = "Mario",
    include_headings: bool = True,
) -> list[DialogueLine]:
    """Parse Markdown content into structured dialogue lines.

    Rules:
        - [Speaker] text → DialogueLine(speaker, text)
        - # / ## / ### Heading → sets heading context for next dialogue line
        - - item / * item → attributed to current speaker
        - Lines without [Speaker] → use last active speaker or default
        - Empty lines → ignored
        - Code blocks → content read as narration by current speaker
        - Markdown formatting (bold, italic, links) → stripped

    Args:
        content: Raw Markdown content string.
        default_speaker: Fallback speaker name for lines without a tag.
        include_headings: If True, heading text is spoken before pauses.
            If False, headings only trigger pauses but aren't read aloud.

    Returns:
        List of DialogueLine with speaker, clean text, and heading metadata.
    """
    lines: list[DialogueLine] = []
    current_speaker = default_speaker
    pending_heading: Optional[str] = None
    pending_heading_level: int = 0
    in_code_block = False
    code_buffer: list[str] = []

    for raw_line in content.splitlines():
        raw_line = raw_line.rstrip()

        # --- Code block handling ---
        if CODE_FENCE_RE.match(raw_line):
            if in_code_block:
                # End of code block
                code_text = "\n".join(code_buffer).strip()
                if code_text:
                    lines.append(
                        DialogueLine(
                            speaker=current_speaker,
                            text=code_text,
                            heading_before=pending_heading,
                            heading_level=pending_heading_level,
                        )
                    )
                    pending_heading = None
                    pending_heading_level = 0
                code_buffer = []
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
                code_buffer = []
            continue

        if in_code_block:
            code_buffer.append(raw_line)
            continue

        stripped = raw_line.strip()

        # --- Empty lines ---
        if not stripped:
            continue

        # --- Headings ---
        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = clean_markdown(heading_match.group(2).strip())
            pending_heading = heading_text
            pending_heading_level = level

            # If include_headings, add the heading as a dialogue line itself
            if include_headings and heading_text:
                lines.append(
                    DialogueLine(
                        speaker=current_speaker,
                        text=heading_text,
                        heading_before=None,
                        heading_level=level,
                    )
                )
                # Reset pending since we already consumed it
                pending_heading = None
                pending_heading_level = 0
            continue

        # --- Speaker tag ---
        speaker_match = SPEAKER_RE.match(stripped)
        if speaker_match:
            current_speaker = speaker_match.group(1).strip()
            text = speaker_match.group(2).strip()
            text = clean_markdown(text)
            if not text:
                continue
            lines.append(
                DialogueLine(
                    speaker=current_speaker,
                    text=text,
                    heading_before=pending_heading,
                    heading_level=pending_heading_level,
                )
            )
            pending_heading = None
            pending_heading_level = 0
            continue

        # --- List items ---
        list_match = LIST_RE.match(stripped)
        if list_match:
            text = clean_markdown(list_match.group(1).strip())
            if not text:
                continue
            lines.append(
                DialogueLine(
                    speaker=current_speaker,
                    text=text,
                    heading_before=pending_heading,
                    heading_level=pending_heading_level,
                    is_list_item=True,
                )
            )
            pending_heading = None
            pending_heading_level = 0
            continue

        # --- Plain text (no speaker, no list) → current speaker ---
        text = clean_markdown(stripped)
        if not text:
            continue
        lines.append(
            DialogueLine(
                speaker=current_speaker,
                text=text,
                heading_before=pending_heading,
                heading_level=pending_heading_level,
            )
        )
        pending_heading = None
        pending_heading_level = 0

    return lines
